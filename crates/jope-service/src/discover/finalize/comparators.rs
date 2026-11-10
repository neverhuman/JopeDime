//! Opt-in same-rows comparator stage of finalization.
//!
//! When a run requested comparators at medium, high or ultra effort, every requested pure-Rust
//! comparator is fitted on exactly the champion's FIT table and scored on exactly its HOLDOUT
//! table with the champion's objective. Fits share one stage deadline and run in a bounded scoped
//! thread pool. A comparator that hits the deadline is reported as timed out and one that errors
//! is reported as failed; neither is dropped and neither changes the champion.

use super::model::predict;
use super::*;
use jope_api::{
    ComparatorBudgetUsed, ComparedModel, ComparedStatus, ComparisonReport, Language,
    ResolvedComparators, ValueProposition,
};
use jope_comparators::{
    fit_comparator, ComparatorError, ComparatorId, ComparatorTier, Deadline, FeatureMatrix,
    FittedComparator,
};
use std::time::{Duration, Instant};

/// `ComparatorStagePlan` is a struct for the finalization comparator stage contract in this
/// module: the resolved tier, roster keys, stage ceiling and the champion's export languages.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComparatorStagePlan {
    pub tier: String,
    pub ids: Vec<String>,
    /// Wall-clock ceiling for the whole stage, shared by every comparator fit.
    pub ceiling_ms: u64,
    pub export_languages: Vec<Language>,
}

impl ComparatorStagePlan {
    /// The stage for an enabled resolution; `None` when comparators are disabled.
    pub fn from_resolved(
        resolved: &ResolvedComparators,
        export_languages: &[Language],
    ) -> Option<Self> {
        if !resolved.enabled {
            return None;
        }
        Some(Self {
            tier: resolved.tier.clone()?,
            ids: resolved.ids.clone(),
            ceiling_ms: resolved.ceiling_seconds?.saturating_mul(1000),
            export_languages: export_languages.to_vec(),
        })
    }

    fn roster(&self) -> Result<(ComparatorTier, Vec<ComparatorId>)> {
        let tier = ComparatorTier::from_effort(&self.tier).context("unknown comparator tier")?;
        let ids = self
            .ids
            .iter()
            .map(|key| ComparatorId::from_key(key).context("unknown comparator id"))
            .collect::<Result<Vec<_>>>()?;
        ensure!(!ids.is_empty(), "comparator stage has no comparators");
        Ok((tier, ids))
    }
}

/// `ChampionSide` is a struct for the champion inputs contract in this module: the FIT-only
/// champion model and its identity, measured with the same holdout procedure as comparators.
pub(super) struct ChampionSide<'a> {
    pub object: &'a ModelObject,
    pub algorithm_id: &'a str,
    pub cv: f64,
}

/// `PartitionDigests` is a struct for the shared-rows identity contract in this module.
pub(super) struct PartitionDigests<'a> {
    pub fit: &'a str,
    pub holdout: &'a str,
}

/// One comparator after the FIT phase: its fitted model and fit time, or its terminal status.
type FitOutcome = std::result::Result<(FittedComparator, f64), ComparedStatus>;

/// Run the comparator stage. FIT rows are read only inside the `ComparatorFit` audit phase and
/// HOLDOUT rows only inside the `ComparatorScore` audit phase.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_stage(
    stage: &ComparatorStagePlan,
    plan: &Plan,
    objective: &RegressionObjective,
    audit: &mut RowAudit,
    fit_rows: &[AuditRowId],
    holdout_rows: &[AuditRowId],
    fit_table: &CsvTable,
    holdout_table: &CsvTable,
    champion: &ChampionSide<'_>,
    digests: &PartitionDigests<'_>,
) -> Result<ComparisonReport> {
    let (tier, ids) = stage.roster()?;
    let started = Instant::now();
    let deadline = Deadline::after(Duration::from_millis(stage.ceiling_ms));
    let fitted = audit.with_rows(RowAuditPhase::ComparatorFit, fit_rows, |_| {
        let (values, targets) = numeric(fit_table, plan)?;
        let features = FeatureMatrix::new(&values, targets.len(), plan.features.len())?;
        fit_all(&ids, tier, features, &targets, plan.seed, &deadline)
    })??;
    let (champion_row, rows) =
        audit.with_rows(RowAuditPhase::ComparatorScore, holdout_rows, |_| {
            score_all(&ids, fitted, plan, objective, holdout_table, champion)
        })??;
    let timed_out = rows
        .iter()
        .filter(|(row, _)| row.status == ComparedStatus::TimedOut)
        .count();
    let value = value_proposition(&champion_row, &rows, &stage.export_languages);
    Ok(ComparisonReport {
        tier: tier.as_str().into(),
        metric: plan.objective.clone(),
        scope: "holdout".into(),
        fit_rows_sha256: digests.fit.into(),
        holdout_rows_sha256: digests.holdout.into(),
        champion: champion_row.0,
        comparators: rows.into_iter().map(|(row, _)| row).collect(),
        value,
        budget: ComparatorBudgetUsed {
            wall_seconds: started.elapsed().as_secs_f64(),
            ceiling_seconds: stage.ceiling_ms as f64 / 1000.0,
            timed_out: u32::try_from(timed_out).unwrap_or(u32::MAX),
        },
    })
}

/// Row-major feature values (plan feature order) and targets from a finalization table.
fn numeric(table: &CsvTable, plan: &Plan) -> Result<(Vec<f64>, Vec<f64>)> {
    let position = |name: &String| {
        table
            .headers
            .iter()
            .position(|header| header == name)
            .context("comparator column missing")
    };
    let columns = plan
        .features
        .iter()
        .map(position)
        .collect::<Result<Vec<_>>>()?;
    let target = position(&plan.target)?;
    let cell = |row: &[Option<String>], column: usize| -> Result<f64> {
        match row.get(column).and_then(Option::as_deref) {
            Some(text) => Ok(text.parse::<f64>()?),
            None => Ok(f64::NAN),
        }
    };
    let mut values = Vec::with_capacity(table.rows.len() * columns.len());
    let mut targets = Vec::with_capacity(table.rows.len());
    for row in &table.rows {
        for &column in &columns {
            values.push(cell(row, column)?);
        }
        targets.push(cell(row, target)?);
    }
    Ok((values, targets))
}

/// Fit every comparator in a bounded scoped pool. Results keep roster order regardless of
/// which worker finished first; each fit is a pure function of its rows, tier and seed.
fn fit_all(
    ids: &[ComparatorId],
    tier: ComparatorTier,
    features: FeatureMatrix<'_>,
    targets: &[f64],
    seed: u64,
    deadline: &Deadline,
) -> Result<Vec<FitOutcome>> {
    let width = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(ids.len())
        .max(1);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let batches = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..width)
            .map(|_| {
                scope.spawn(|| {
                    let mut done = Vec::new();
                    loop {
                        let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(&id) = ids.get(index) else { break };
                        let clock = Instant::now();
                        let outcome = fit_comparator(id, tier, features, targets, seed, deadline)
                            .map(|model| (model, clock.elapsed().as_secs_f64()))
                            .map_err(status_of);
                        done.push((index, outcome));
                    }
                    done
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|worker| worker.join())
            .collect::<std::result::Result<Vec<_>, _>>()
    })
    .map_err(|_| anyhow::anyhow!("comparator worker stopped unexpectedly"))?;
    let mut outcomes: Vec<Option<FitOutcome>> = ids.iter().map(|_| None).collect();
    for (index, outcome) in batches.into_iter().flatten() {
        let slot = outcomes.get_mut(index).context("comparator result index")?;
        *slot = Some(outcome);
    }
    outcomes
        .into_iter()
        .map(|outcome| outcome.context("comparator result missing"))
        .collect()
}

fn status_of(error: ComparatorError) -> ComparedStatus {
    match error {
        ComparatorError::DeadlineExceeded => ComparedStatus::TimedOut,
        other => ComparedStatus::Failed {
            reason: other.to_string(),
        },
    }
}

/// A comparison row plus its objective loss (lower is better) when it completed.
type ScoredRow = (ComparedModel, Option<f64>);

/// Score the champion and every fitted comparator on the HOLDOUT table.
fn score_all(
    ids: &[ComparatorId],
    fitted: Vec<FitOutcome>,
    plan: &Plan,
    objective: &RegressionObjective,
    holdout: &CsvTable,
    champion: &ChampionSide<'_>,
) -> Result<(ScoredRow, Vec<ScoredRow>)> {
    let (values, targets) = numeric(holdout, plan)?;
    let features = FeatureMatrix::new(&values, targets.len(), plan.features.len())?;
    let rows = targets.len().max(1) as f64;
    let clock = Instant::now();
    let predictions = predict(champion.object, holdout)?;
    let champion_us = clock.elapsed().as_secs_f64() * 1e6 / rows;
    let champion_score = objective.evaluate(&targets, &predictions)?;
    let champion_bytes = std::iter::once(champion.object.manifest_bytes().len())
        .chain(champion.object.parts().values().map(Vec::len))
        .sum::<usize>() as u64;
    let champion_row = ComparedModel {
        id: champion.algorithm_id.into(),
        label: format!("JopeDime champion ({})", champion.algorithm_id),
        implementation: "jope_champion".into(),
        status: ComparedStatus::Completed,
        holdout: Some(champion_score.value),
        cv: Some(champion.cv),
        fit_seconds: None,
        predict_us_per_row: Some(champion_us),
        model_bytes: Some(champion_bytes),
    };
    let scored = ids
        .iter()
        .zip(fitted)
        .map(|(&id, outcome)| {
            let descriptor = id.descriptor();
            let mut row = ComparedModel {
                id: descriptor.key.into(),
                label: descriptor.label.into(),
                implementation: "rust_port_analogue".into(),
                status: ComparedStatus::Completed,
                holdout: None,
                cv: None,
                fit_seconds: None,
                predict_us_per_row: None,
                model_bytes: None,
            };
            let (model, fit_seconds) = match outcome {
                Ok(fitted) => fitted,
                Err(status) => {
                    row.status = status;
                    return (row, None);
                }
            };
            row.fit_seconds = Some(fit_seconds);
            row.model_bytes = Some(model.model_bytes());
            let clock = Instant::now();
            let measured = model
                .predict(features)
                .map_err(anyhow::Error::from)
                .map(|predictions| (clock.elapsed().as_secs_f64() * 1e6 / rows, predictions))
                .and_then(|(us, predictions)| {
                    Ok((us, objective.evaluate(&targets, &predictions)?))
                });
            match measured {
                Ok((us, score)) => {
                    row.predict_us_per_row = Some(us);
                    row.holdout = Some(score.value);
                    (row, Some(score.loss))
                }
                Err(error) => {
                    row.status = ComparedStatus::Failed {
                        reason: error.to_string(),
                    };
                    (row, None)
                }
            }
        })
        .collect();
    Ok(((champion_row, Some(champion_score.loss)), scored))
}

/// Summarize the champion against completed comparators. Positive deltas mean the champion's
/// objective loss is lower; a loss is reported as a loss, never hidden.
pub(super) fn value_proposition(
    champion: &ScoredRow,
    rows: &[ScoredRow],
    export_languages: &[Language],
) -> ValueProposition {
    let ours = champion.1.unwrap_or(f64::INFINITY);
    let completed: Vec<(&ComparedModel, f64)> = rows
        .iter()
        .filter_map(|(row, loss)| loss.map(|loss| (row, loss)))
        .collect();
    let best = completed
        .iter()
        .min_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.id.cmp(&b.0.id)));
    let mut losses: Vec<f64> = completed.iter().map(|(_, loss)| *loss).collect();
    losses.sort_by(f64::total_cmp);
    let median = match losses.len() {
        0 => None,
        n if n % 2 == 1 => losses.get(n / 2).copied(),
        n => losses
            .get(n / 2 - 1)
            .zip(losses.get(n / 2))
            .map(|(a, b)| (a + b) / 2.0),
    };
    let delta = |other: f64| (other != 0.0).then(|| (other - ours) / other.abs() * 100.0);
    let count = |keep: &dyn Fn(f64) -> bool| {
        u32::try_from(completed.iter().filter(|(_, loss)| keep(*loss)).count()).unwrap_or(u32::MAX)
    };
    let loses_to = count(&|loss| loss < ours);
    let ratio = |numerator: Option<f64>, denominator: Option<f64>| {
        numerator
            .zip(denominator)
            .filter(|(_, d)| *d > 0.0)
            .map(|(n, d)| n / d)
    };
    ValueProposition {
        rank: loses_to.saturating_add(1),
        entrants: u32::try_from(completed.len() + 1).unwrap_or(u32::MAX),
        best_alternative: best.map(|(row, _)| row.id.clone()),
        holdout_delta_pct_vs_best: best.and_then(|(_, loss)| delta(*loss)),
        holdout_delta_pct_vs_median: median.and_then(delta),
        beats: count(&|loss| loss > ours),
        loses_to,
        predict_speedup_vs_best: best
            .and_then(|(row, _)| ratio(row.predict_us_per_row, champion.0.predict_us_per_row)),
        size_ratio_vs_best: best.and_then(|(row, _)| {
            ratio(
                champion.0.model_bytes.map(|b| b as f64),
                row.model_bytes.map(|b| b as f64),
            )
        }),
        export_languages: export_languages.to_vec(),
        deterministic_replay: true,
    }
}
