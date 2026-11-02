use super::helpers::{field, float_field, mean, optional_number, score_rank_percentile, set};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;

/// The gate's deployed-score map and generation history.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GateState {
    pub self_best: BTreeMap<String, f64>,
    pub gen_top1: BTreeMap<u64, BTreeMap<String, f64>>,
    pub elo_history: Vec<Value>,
    pub last_shipped_deployed: Option<BTreeMap<String, f64>>,
    pub last_shipped_generation: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GateSummary {
    pub beat_rate_vs_prev: Option<f64>,
    pub gate_baseline_generation: Option<u64>,
    pub compared: usize,
    pub new_bests: usize,
    pub cum_best_datasets: usize,
    pub mean_rank_pct: Option<f64>,
    pub shipped: bool,
}

/// Port of `gate`: exact score ties count as half a win; missing comparisons do
/// not reject a generation, and self-best state rises monotonically.
pub fn gate_generation(
    state: &mut GateState,
    generation: u64,
    deployed: &BTreeMap<String, f64>,
    gate_beat_rate: f64,
    score_pool: &BTreeMap<String, Vec<f64>>,
) -> GateSummary {
    let (previous, baseline_generation) = last_shipped_deployed_scores(state, generation);
    let mut beats = 0.0;
    let mut compared = 0;
    for (dataset, score) in deployed {
        if let Some(previous_score) = previous.get(dataset) {
            compared += 1;
            if score > previous_score {
                beats += 1.0;
            } else if score == previous_score {
                beats += 0.5;
            }
        }
    }
    let beat_rate = (compared > 0).then(|| beats / compared as f64);
    let mut new_bests = 0;
    for (dataset, score) in deployed {
        if state.self_best.get(dataset).is_none_or(|best| score > best) {
            state.self_best.insert(dataset.clone(), *score);
            new_bests += 1;
        }
    }
    let mean_rank_pct = mean(
        &deployed
            .iter()
            .map(|(dataset, score)| {
                let mut pool = score_pool.get(dataset).cloned().unwrap_or_default();
                score_rank_percentile(&mut pool, *score)
            })
            .collect::<Vec<_>>(),
    );
    state.gen_top1.insert(generation, deployed.clone());
    let shipped = beat_rate.is_none_or(|rate| rate >= gate_beat_rate);
    GateSummary {
        beat_rate_vs_prev: beat_rate,
        gate_baseline_generation: baseline_generation,
        compared,
        new_bests,
        cum_best_datasets: state.self_best.len(),
        mean_rank_pct,
        shipped,
    }
}

/// Implements the explicit, historical and legacy fallback order in the Python mixin.
pub fn last_shipped_deployed_scores(
    state: &GateState,
    generation: u64,
) -> (BTreeMap<String, f64>, Option<u64>) {
    if let Some(explicit) = &state.last_shipped_deployed {
        return (explicit.clone(), state.last_shipped_generation);
    }
    let mut shipped: Vec<(u64, &Value)> = state
        .elo_history
        .iter()
        .filter(|row| {
            field(row, "shipped")
                .and_then(Value::as_bool)
                .unwrap_or(true)
        })
        .filter_map(|row| {
            let gen = field(row, "gen")?.as_u64()?;
            (gen < generation).then_some((gen, row))
        })
        .collect();
    shipped.sort_by_key(|(gen, _)| std::cmp::Reverse(*gen));
    for (gen, _) in shipped {
        if let Some(scores) = state.gen_top1.get(&gen) {
            return (scores.clone(), Some(gen));
        }
    }
    state
        .gen_top1
        .get(&generation.saturating_sub(1))
        .map(|scores| (scores.clone(), Some(generation.saturating_sub(1))))
        .unwrap_or_default()
}

/// Selects the canonical deployment Elo (probe, then CV, then legacy top1).
pub fn deployment_elo(row: &Value) -> Option<f64> {
    ["elo_probe", "elo_cv", "elo_top1"]
        .into_iter()
        .find_map(|key| float_field(row, key))
}

/// Port of the Python Elo override to the per-dataset beat gate.
pub fn resolve_ship_decision(gate_shipped: bool, elo: &Value, history: &[Value]) -> Value {
    let prior: Vec<f64> = history
        .iter()
        .filter(|row| {
            field(row, "shipped")
                .and_then(Value::as_bool)
                .unwrap_or(false)
        })
        .filter_map(deployment_elo)
        .collect();
    let current = deployment_elo(elo);
    let baseline = prior.into_iter().max_by(f64::total_cmp);
    let record = current.is_some_and(|value| baseline.is_none_or(|best| value >= best));
    let regressed = current.is_some_and(|value| baseline.is_some_and(|best| value < best));
    json!({
        "shipped": (gate_shipped && !regressed) || record,
        "elo_record": record,
        "deployed_elo_baseline": optional_number(baseline),
        "deployed_elo_delta": current.zip(baseline).map(|(value, best)| value - best),
        "deployed_elo_regressed": regressed,
    })
}

/// Flattens the benchmark report exactly as `bench` does and derives Elo regrets.
pub fn benchmark_summary(report: &Value) -> Option<Value> {
    let modes = field(report, "modes")?;
    let top1 = field(modes, "top1")?;
    let cv_select = field(modes, "cv_select")?;
    let best_k = field(modes, "best_k")?;
    let probe = field(modes, "cv_top6").unwrap_or(cv_select);
    let mut out = json!({
        "elo_top1": field(top1, "elo")?,
        "elo_rank": field(top1, "field_rank")?,
        "elo_top1_dh": field(top1, "double_holdout_elo").cloned().unwrap_or(Value::Null),
        "elo_probe": field(probe, "elo")?,
        "elo_probe_rank": field(probe, "field_rank")?,
        "elo_probe_dh": field(probe, "double_holdout_elo").cloned().unwrap_or(Value::Null),
        "battles_won": field(probe, "battles_won").cloned().unwrap_or(Value::Null),
        "battles_top3": field(probe, "battles_top3").cloned().unwrap_or(Value::Null),
        "elo_cv": field(cv_select, "elo")?,
        "elo_best_k": field(best_k, "elo")?,
    });
    let deployed = deployment_elo(&out);
    set(&mut out, "elo_deployed", optional_number(deployed));
    let best = float_field(&out, "elo_best_k");
    let top = float_field(&out, "elo_top1");
    let probed = float_field(&out, "elo_probe");
    if let (Some(best), Some(top)) = (best, top) {
        set(&mut out, "top1_regret_vs_best_k", json!(best - top));
    }
    if let (Some(best), Some(probed)) = (best, probed) {
        set(&mut out, "probe_regret_vs_best_k", json!(best - probed));
    }
    Some(out)
}
