use super::helpers::{
    choose_deployed_candidate, field, finite, float_field, mean, optional_number, robust_score,
    round_six, score_rank_percentile, set,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// Evaluation input row, matching the private keys consumed by the Python mixin.
#[derive(Clone, Debug, PartialEq)]
pub struct EvaluationInput {
    pub dataset_hash: String,
    pub rank: usize,
    pub engine: String,
    pub genome: String,
    pub candidate_pool_id: Option<String>,
}

/// Candidate rows are kept in insertion order, as Python dictionaries are.
#[derive(Clone, Debug, PartialEq)]
pub struct DatasetCandidates {
    pub dataset_hash: String,
    pub rows: Vec<Value>,
}

/// Dataset-derived vectors already loaded by the caller from its `.vs` artifacts.
#[derive(Clone, Debug, PartialEq)]
pub struct DatasetContext {
    pub embedding: Vec<f64>,
    pub target_embedding: Vec<f64>,
    pub former_best_score: Option<f64>,
}

/// Results and state required for one `evaluate_and_append` pass.
#[derive(Clone, Debug)]
pub struct EvaluationRequest {
    pub generation: u64,
    pub run_name: String,
    pub artifact_dir: PathBuf,
    pub run_dir: PathBuf,
    pub candidate_pool_dir: Option<PathBuf>,
    pub eval_inputs: Vec<EvaluationInput>,
    pub eval_results: Vec<(usize, Value)>,
    pub picks: Vec<DatasetCandidates>,
    pub datasets: BTreeMap<String, DatasetContext>,
    pub candidate_pool_rows: Vec<Value>,
    /// Existing historical scores, in any order; rows appended here are sorted.
    pub score_pool: BTreeMap<String, Vec<f64>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvaluationSummary {
    pub rows: usize,
    pub deployed_scores: BTreeMap<String, f64>,
    pub self_rows_path: String,
    pub candidate_pool_path: Option<String>,
    pub selector_regret_at_1: Option<f64>,
    pub capture_at_6: Option<f64>,
    pub rows_out: Vec<Value>,
    pub candidate_pool_rows: Vec<Value>,
    pub scored_events: Vec<ScoredEvent>,
    /// Evaluation progress checkpoints (done,total), including the final row.
    pub progress: Vec<(usize, usize)>,
    pub score_pool: BTreeMap<String, Vec<f64>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScoredEvent {
    pub dataset_hash: String,
    pub engine: String,
    pub genome: String,
    pub oof_r2: Option<f64>,
    pub cache_hit: bool,
}

/// Port of `SelfPlayEvaluationMixin.evaluate_and_append`'s deterministic data path.
/// Results are supplied as indexed values after the external ledger evaluator runs.
pub fn evaluate_and_append(mut request: EvaluationRequest) -> std::io::Result<EvaluationSummary> {
    let result_by_index: BTreeMap<usize, Value> = request.eval_results.iter().cloned().collect();
    let mut pool_index = BTreeMap::<String, usize>::new();
    for (index, row) in request.candidate_pool_rows.iter().enumerate() {
        if let Some(id) = field(row, "candidate_pool_id").and_then(Value::as_str) {
            pool_index.insert(id.to_owned(), index);
        }
    }

    let mut scored_events = Vec::new();
    let mut progress = Vec::new();
    let mut done = 0;
    for (index, result) in &request.eval_results {
        let Some(input) = request.eval_inputs.get(*index) else {
            continue;
        };
        done += 1;
        let cv = field(result, "cv").filter(|value| value.is_object());
        let oof = finite(cv.and_then(|value| field(value, "oof_r2")));
        scored_events.push(ScoredEvent {
            dataset_hash: input.dataset_hash.clone(),
            engine: input.engine.clone(),
            genome: input.genome.clone(),
            oof_r2: oof,
            cache_hit: field(result, "cache_hit")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        });
        if done % 25 == 0 || done == request.eval_inputs.len() {
            progress.push((done, request.eval_inputs.len()));
        }
    }

    // Pass 1: attach the robust score and raw evaluation measurements to the
    // selected candidate and its candidate-pool row, matching Python's order.
    for (index, input) in request.eval_inputs.iter().enumerate() {
        let result = result_by_index.get(&index).cloned().unwrap_or(Value::Null);
        let cv = field(&result, "cv").filter(|value| value.is_object());
        let oof = finite(cv.and_then(|value| field(value, "oof_r2")));
        let fold_values = cv
            .and_then(|value| field(value, "fold_r2"))
            .and_then(Value::as_array)
            .map(|values| values.iter().filter_map(Value::as_f64).collect::<Vec<_>>());
        let cost_ms = field(&result, "timings")
            .and_then(|timings| field(timings, "total_ms"))
            .cloned()
            .unwrap_or(Value::Null);
        if let Some(group) = request
            .picks
            .iter_mut()
            .find(|group| group.dataset_hash == input.dataset_hash)
        {
            if let Some(record) = group.rows.get_mut(input.rank) {
                if let Some(oof) = oof {
                    let score = robust_score(Some(oof), fold_values.as_deref());
                    set(record, "score_robust", optional_number(score));
                    set(record, "_oof", json!(oof));
                    set(record, "_cost_ms", cost_ms.clone());
                    if let Some(pool_id) = input.candidate_pool_id.as_deref() {
                        if let Some(pool_row) = pool_index.get(pool_id).copied() {
                            set(
                                &mut request.candidate_pool_rows[pool_row],
                                "measured_oof",
                                json!(oof),
                            );
                            set(
                                &mut request.candidate_pool_rows[pool_row],
                                "score_robust",
                                optional_number(score),
                            );
                            set(
                                &mut request.candidate_pool_rows[pool_row],
                                "cost_ms",
                                cost_ms,
                            );
                        }
                    }
                } else {
                    set(record, "score_robust", Value::Null);
                }
            }
        }
    }

    let mut deployed_scores = BTreeMap::new();
    let mut regrets = Vec::new();
    let mut captures_at_6 = Vec::new();
    let mut rows_out = Vec::new();
    let mut score_pool = request.score_pool;

    for picks in &mut request.picks {
        for row in &mut picks.rows {
            set(row, "deployed", Value::Bool(false));
        }
        let Some(best_index) = choose_deployed_candidate(&picks.rows) else {
            continue;
        };
        set(&mut picks.rows[best_index], "deployed", Value::Bool(true));
        let best = picks.rows[best_index].clone();
        let best_score = float_field(&best, "score_robust").expect("selected scored row");
        deployed_scores.insert(picks.dataset_hash.clone(), best_score);

        let measured: Vec<&Value> = picks
            .rows
            .iter()
            .filter(|row| float_field(row, "score_robust").is_some())
            .collect();
        let mut best_measured: Option<&Value> = None;
        for row in &measured {
            if best_measured.is_none_or(|best_row| {
                float_field(row, "score_robust")
                    .unwrap()
                    .total_cmp(&float_field(best_row, "score_robust").unwrap())
                    == std::cmp::Ordering::Greater
            }) {
                best_measured = Some(row);
            }
        }
        if let Some(best_measured) = best_measured {
            let measured_score = float_field(best_measured, "score_robust").unwrap_or(0.0);
            let selected_score = float_field(&best, "score_robust").unwrap_or(0.0);
            regrets.push(measured_score - selected_score);
            captures_at_6.push(
                field(best_measured, "rank")
                    .and_then(Value::as_i64)
                    .unwrap_or(0)
                    < 6,
            );
        }

        if let Some(id) = field(&best, "candidate_pool_id").and_then(Value::as_str) {
            if let Some(index) = pool_index.get(id).copied() {
                set(
                    &mut request.candidate_pool_rows[index],
                    "deployed",
                    Value::Bool(true),
                );
            }
        }
    }

    // Python emits training rows by picks/dataset insertion order, then each
    // dataset's candidate order. Rank order is retained in each candidate vector.
    for picks in &request.picks {
        let context = request.datasets.get(&picks.dataset_hash);
        let former_best = context.and_then(|value| value.former_best_score);
        for rec in &picks.rows {
            let Some(oof) = float_field(rec, "_oof") else {
                continue;
            };
            let Some(score) = float_field(rec, "score_robust") else {
                continue;
            };
            let context = context.expect("dataset context is required for scored candidates");
            let rank = field(rec, "rank").cloned().unwrap_or(Value::Null);
            let candidate_pool_id = field(rec, "candidate_pool_id")
                .cloned()
                .unwrap_or(Value::Null);
            let rank_pct = score_rank_percentile(
                score_pool.entry(picks.dataset_hash.clone()).or_default(),
                score,
            );
            let metadata = json!({
                "provenance": format!("self_g{}", request.generation),
                "generation": request.generation,
                "deployed": field(rec, "deployed").and_then(Value::as_bool).unwrap_or(false),
                "task_type": "regression",
                "split": "train",
                "oof_r2_raw": oof,
                "score_robust": score,
                "rank_pct": round_six(rank_pct),
                "former_best_score": optional_number(former_best),
                "candidate_pool_id": candidate_pool_id,
                "selected_for_eval": true,
                "selected_for_eval_rank": rank,
                "reward": field(rec, "reward").cloned().unwrap_or(Value::Null),
                "reward_std": field(rec, "reward_std").cloned().unwrap_or(Value::Null),
                "reward_percentile_in_engine": field(rec, "reward_percentile_in_engine").cloned().unwrap_or(Value::Null),
                "p_bad": field(rec, "p_bad").cloned().unwrap_or(Value::Null),
                "selection_key": field(rec, "selection_key").cloned().unwrap_or(Value::Null),
                "selection_score": field(rec, "selection_score").cloned().unwrap_or(Value::Null),
                "selector_reranker_score": field(rec, "selector_reranker_score").cloned().unwrap_or(Value::Null),
            });
            rows_out.push(json!({
                "dataset_id": picks.dataset_hash,
                "source": "self_play",
                "embedding": context.embedding,
                "target_embedding": context.target_embedding,
                "genome": field(rec, "genome").cloned().unwrap_or(Value::String(String::new())),
                "canonical_genome": field(rec, "genome").cloned().unwrap_or(Value::String(String::new())),
                "valid": true,
                "metric": Value::Null,
                "score": oof.max(-1.0),
                "rank": Value::Null,
                "elo": Value::Null,
                "cost_ms": field(rec, "_cost_ms").cloned().unwrap_or(Value::Null),
                "family_bitmap": {},
                "complexity": Value::Null,
                "runtime_class": Value::Null,
                "feedback_origin": Value::Null,
                "engine": field(rec, "engine").cloned().unwrap_or(Value::String("featmath".into())),
                "metadata": metadata,
            }));
            let scores = score_pool.entry(picks.dataset_hash.clone()).or_default();
            scores.push(score);
            scores.sort_by(f64::total_cmp);
        }
    }

    let rows_path = request.artifact_dir.join(format!(
        "self_rows_{}_g{}.jsonl",
        request.run_name, request.generation
    ));
    write_jsonl(&rows_path, &rows_out)?;
    let candidate_pool_path = if let Some(directory) = request.candidate_pool_dir.as_ref() {
        let path = directory.join(format!("candidate_pools_g{}.jsonl", request.generation));
        write_jsonl(&path, &request.candidate_pool_rows)?;
        Some(path.to_string_lossy().into_owned())
    } else {
        None
    };

    let selector_regret_at_1 = mean(&regrets);
    let capture_at_6 = mean(
        &captures_at_6
            .iter()
            .map(|value| if *value { 1.0 } else { 0.0 })
            .collect::<Vec<_>>(),
    );
    Ok(EvaluationSummary {
        rows: rows_out.len(),
        deployed_scores,
        self_rows_path: rows_path.to_string_lossy().into_owned(),
        candidate_pool_path,
        selector_regret_at_1,
        capture_at_6,
        rows_out,
        candidate_pool_rows: request.candidate_pool_rows,
        scored_events,
        progress,
        score_pool,
    })
}

fn write_jsonl(path: &Path, rows: &[Value]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut writer = BufWriter::new(File::create(path)?);
    for row in rows {
        serde_json::to_writer(&mut writer, row)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()
}
