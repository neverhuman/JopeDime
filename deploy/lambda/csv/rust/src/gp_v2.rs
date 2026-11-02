//! Isolated, bounded Lambda GP screening. Never canonical benchmark/learning admission.
#[cfg(test)]
#[macro_use]
#[path = "gp_projection_vocabulary_audit_test.rs"]
mod gp_projection_vocabulary_audit_test;
use feat_math::{
    dsl::{ParamValue, Stage},
    evolve, BoundModel, ColumnSpec, DataSchema, GenomeIr, Objective, RawTable, RowSemantics,
    TaskKind, TaskSpec,
};
use lambda_runtime::{service_fn, Error, LambdaEvent};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
mod gp_v2_setup;
use gp_v2_setup::*;
const MANIFEST: &str = include_str!(env!("JOPE_GP_MANIFEST_PATH"));
const PROTOCOL: &str = "jope.lambda-gp-screen.v1";
const DATASET_BUCKET: &str = "veox-jopedime-datasets-use2";
const LEARNING_BUCKET: &str = "veox-jopedime-learning-use2";
const CURATED_PREFIX: &str = "experiments/lambda-gp/v1/campaigns/gp-corpus-20260928-v1/";
fn default_seconds() -> u64 {
    25
}
fn hash(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn jobs() -> Result<Vec<Job>, Error> {
    Ok(serde_json::from_str(MANIFEST)?)
}
fn memory() -> Value {
    let s = fs::read_to_string("/proc/self/status").unwrap_or_default();
    let get = |key: &str| -> Option<u64> {
        s.lines()
            .find(|l| l.starts_with(key))?
            .split_whitespace()
            .nth(1)?
            .parse()
            .ok()
    };
    let stat = fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let fields: Vec<&str> = stat
        .rsplit_once(')')
        .map(|(_, s)| s.split_whitespace().collect())
        .unwrap_or_default();
    let tick = |index: usize| fields.get(index).and_then(|v| v.parse::<u64>().ok());
    json!({"rss_kib":get("VmRSS:"), "process_lifetime_hwm_kib":get("VmHWM:"),"process_user_clock_ticks":tick(11),"process_system_clock_ticks":tick(12)})
}
fn url_encode(s: &str) -> String {
    s.bytes()
        .map(|b| {
            if b.is_ascii_alphanumeric() || b"-_.~".contains(&b) {
                (b as char).to_string()
            } else {
                format!("%{b:02X}")
            }
        })
        .collect()
}
fn download(agent: &ureq::Agent, job: &Job, url: &str) -> Result<Vec<u8>, Error> {
    let (path, query) = url.split_once('?').ok_or("missing presigned query")?;
    if path
        != format!(
            "https://{}.s3.us-east-2.amazonaws.com/{}",
            job.bucket, job.key
        )
        || job.key.contains("..")
        || job.key.contains(['?', '#', '%'])
    {
        return Err("download URL does not match admitted S3 key".into());
    }
    let expected = format!("versionId={}", url_encode(&job.version));
    if !query.split('&').any(|v| v == expected)
        || query
            .split('&')
            .filter(|v| v.starts_with("versionId="))
            .count()
            != 1
    {
        return Err("download URL does not match admitted version".into());
    }
    // The role has no S3 privileges. This short-lived URL is never returned or logged.
    let mut response = agent
        .get(url)
        .call()
        .map_err(|_| "S3 download failed; URL redacted")?;
    if response
        .headers()
        .get("x-amz-version-id")
        .and_then(|v| v.to_str().ok())
        != Some(job.version.as_str())
    {
        return Err("S3 response version mismatch".into());
    }
    Ok(response
        .body_mut()
        .with_config()
        .limit(job.bytes as u64 + 1)
        .read_to_vec()
        .map_err(|_| "S3 response exceeded admitted bytes or transport failed")?)
}
fn admit(job: &Job) -> Result<(), Error> {
    if !((job.bucket == DATASET_BUCKET && job.key.starts_with("blobs/sha256/"))
        || (job.bucket == LEARNING_BUCKET && job.key.starts_with(CURATED_PREFIX)))
    {
        return Err("bucket/key outside pinned input namespaces".into());
    }
    let sealed = job.partition == "meta_test"
        || job.partition == "sealed"
        || job.metadata.get("sealed").and_then(Value::as_bool) == Some(true)
        || job
            .metadata
            .get("historical_partitions")
            .and_then(Value::as_array)
            .is_some_and(|xs| xs.iter().any(|v| v == "meta_test" || v == "sealed"));
    if sealed {
        return Err("sealed family refused before download".into());
    }
    Ok(())
}
fn remaining_ms(deadline: u64) -> u64 {
    deadline.saturating_sub(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64,
    )
}
#[derive(Clone, Serialize)]
struct Fold {
    fit: Vec<usize>,
    selection: Vec<usize>,
    audit: Vec<usize>,
}
fn fold(n: usize, dataset: &str, seed: u64, ordered: bool) -> Fold {
    let mut order: Vec<usize> = (0..n).collect();
    if !ordered {
        order.sort_by_cached_key(|i| {
            Sha256::digest(format!("{PROTOCOL}:{dataset}:{seed}:{i}")).to_vec()
        });
    }
    let fit_n = (n * 3 / 5).max(2);
    let selection_n = (n / 5).max(1);
    let mut result = Fold {
        fit: order[..fit_n].to_vec(),
        selection: order[fit_n..fit_n + selection_n].to_vec(),
        audit: order[fit_n + selection_n..].to_vec(),
    };
    result.fit.sort_unstable();
    result.selection.sort_unstable();
    result.audit.sort_unstable();
    result
}
struct Prepared {
    fit: RawTable,
    selection: RawTable,
    audit: RawTable,
    fit_y: Vec<f64>,
    selection_y: Vec<f64>,
    audit_y: Vec<f64>,
    fold: Fold,
    schema: DataSchema,
    task: TaskSpec,
    labels: Vec<f64>,
    columns: Vec<usize>,
    original_features: usize,
}
fn number(field: &str) -> Result<Option<f64>, Error> {
    let field = field.trim();
    let f = if field.starts_with('"') && field.ends_with('"') && field.len() >= 2 {
        &field[1..field.len() - 1]
    } else {
        field
    };
    if f.contains('"') || (field.starts_with('"') != field.ends_with('"')) {
        return Err("malformed CSV quote".into());
    }
    if f.is_empty() || matches!(f, "?" | "NA" | "NaN" | "nan" | "null") {
        return Ok(None);
    }
    let n: f64 = f
        .parse()
        .map_err(|_| "non-numeric cell unsupported by numeric corpus protocol")?;
    if !n.is_finite() {
        return Err("nonfinite numeric cell".into());
    }
    Ok(Some(n))
}
fn prepare(raw: &[u8], job: &Job, seed: u64) -> Result<Prepared, Error> {
    if raw.len() != job.bytes || hash(raw) != job.sha256 {
        return Err("CSV SHA256/byte mismatch".into());
    }
    if job.bytes > MAX_BYTES
        || job.rows < 5
        || job.rows > MAX_ROWS
        || job.columns < 2
        || job.columns > MAX_COLS
    {
        return Err(
            "unsupported shape: need 5..50000 rows, 2..21000 columns, <=150MB; rows never sampled"
                .into(),
        );
    }
    admit(job)?;
    let class = matches!(job.task.as_str(), "binary" | "multiclass" | "multi_label");
    let task = match job.task.as_str() {
        "regression" => TaskSpec {
            kind: TaskKind::Regression,
            objective: Objective::Rmse,
        },
        "binary" => TaskSpec {
            kind: TaskKind::Binary,
            objective: Objective::LogLoss,
        },
        "multi_label" | "multiclass" => TaskSpec {
            kind: TaskKind::Multiclass,
            objective: Objective::MacroF1,
        },
        // The corpus lacks original timestamps; declared CSV order is the only available chronology.
        "timeseries" => TaskSpec {
            kind: TaskKind::Regression,
            objective: Objective::Rmse,
        },
        _ => return Err("unsupported original task".into()),
    };
    let fold = fold(job.rows, &job.dataset_hash, seed, job.task == "timeseries");
    let p = job.columns - 1;
    let mut n = vec![0u64; p];
    let mut means = vec![0.0; p];
    let mut m2 = vec![0.0; p];
    let mut y = Vec::with_capacity(job.rows);
    let text = std::str::from_utf8(raw)?;
    for (i, line) in text.lines().enumerate() {
        if i >= job.rows {
            return Err("CSV row count mismatch".into());
        }
        let mut count = 0;
        let is_fit = fold.fit.binary_search(&i).is_ok();
        for (j, cell) in line.split(',').enumerate() {
            if j >= job.columns {
                return Err("CSV column count mismatch".into());
            }
            let value = number(cell)?;
            if j == p {
                y.push(value.ok_or("missing target")?);
            } else if is_fit {
                if let Some(v) = value {
                    n[j] += 1;
                    let delta = v - means[j];
                    means[j] += delta / n[j] as f64;
                    m2[j] += delta * (v - means[j]);
                }
            }
            count += 1;
        }
        if count != job.columns {
            return Err("CSV column count mismatch".into());
        }
    }
    if y.len() != job.rows {
        return Err("CSV row count mismatch".into());
    }
    let mut columns: Vec<usize> = (0..p).collect();
    if p > WIDTH {
        columns.sort_by(|&a, &b| {
            let variance = |j: usize| if n[j] > 1 { m2[j] / n[j] as f64 } else { 0.0 };
            variance(b).total_cmp(&variance(a)).then(a.cmp(&b))
        });
        columns.truncate(WIDTH);
        columns.sort_unstable();
    }
    let mut labels = Vec::new();
    if class {
        // Label vocabulary is learned only from fit rows. Unseen selection/audit labels keep -1,
        // count as errors, and never influence fitting or the vocabulary.
        labels = fold.fit.iter().map(|&i| y[i]).collect();
        labels.sort_by(f64::total_cmp);
        labels.dedup();
        if labels.len() < 2 || labels.len() > 1000 || (job.task == "binary" && labels.len() != 2) {
            return Err("classification fit partition has unsupported class count".into());
        }
        for value in &mut y {
            *value = labels
                .binary_search_by(|v| v.total_cmp(value))
                .map(|i| i as f64)
                .unwrap_or(-1.0);
        }
    }
    let names: Vec<String> = columns
        .iter()
        .map(|j| format!("original_column_{j}"))
        .collect();
    let schema = DataSchema {
        columns: names.iter().map(ColumnSpec::numeric).collect(),
        row_semantics: RowSemantics::Iid,
    };
    let mut selected = vec![false; p];
    for &j in &columns {
        selected[j] = true;
    }
    let mut fit_rows = Vec::new();
    let mut selection_rows = Vec::new();
    let mut audit_rows = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let row: Vec<String> = line
            .split(',')
            .enumerate()
            .filter(|(j, _)| *j < p && selected[*j])
            .map(|(_, v)| number(v).map(|v| v.map(|n| n.to_string()).unwrap_or_default()))
            .collect::<Result<_, _>>()?;
        if fold.fit.binary_search(&i).is_ok() {
            fit_rows.push(row);
        } else if fold.selection.binary_search(&i).is_ok() {
            selection_rows.push(row);
        } else {
            audit_rows.push(row);
        }
    }
    Ok(Prepared {
        fit: RawTable::new(names.clone(), fit_rows)?,
        selection: RawTable::new(names.clone(), selection_rows)?,
        audit: RawTable::new(names, audit_rows)?,
        fit_y: fold.fit.iter().map(|&i| y[i]).collect(),
        selection_y: fold.selection.iter().map(|&i| y[i]).collect(),
        audit_y: fold.audit.iter().map(|&i| y[i]).collect(),
        fold,
        schema,
        task,
        labels,
        columns,
        original_features: p,
    })
}
const TRANSFORMS: &[&str] = &[
    "standardize",
    "robust_scale",
    "quantile_bins",
    "winsor_clip",
    "sigmoid",
    "symlog",
];
// Rejection sampling of the existing engine's grammar, not an unbounded evaluator.
// Heads are ridge-family only; actual bound task heads are fitted, never surrogate scored.
fn safe_genome(source: &str) -> Result<String, Error> {
    let GenomeIr::Pipeline(mut dsl) = GenomeIr::parse(source)? else {
        return Err("pipeline required".into());
    };
    if dsl.stages.is_empty() || dsl.stages.len() > 3 {
        return Err("stage bound".into());
    }
    let mut head = dsl.stages.pop().unwrap();
    if head.op != "ridge" {
        return Err("head outside bounded ridge-family grammar".into());
    }
    for stage in &dsl.stages {
        if !TRANSFORMS.contains(&stage.op.as_str())
            || matches!(stage.params.get("append"), Some(ParamValue::Bool(true)))
        {
            return Err("transform outside bounded width-preserving grammar".into());
        }
    }
    head.params.retain(|k, _| k == "lambda_frac");
    let lambda = head.get_f64("lambda_frac", 0.02).clamp(0.00001, 1.0);
    head.params
        .insert("lambda_frac".into(), ParamValue::Float(lambda));
    dsl.stages.push(head);
    Ok(GenomeIr::Pipeline(dsl).canonical())
}
fn bound_genome(source: &str, task: TaskKind) -> Result<GenomeIr, Error> {
    let GenomeIr::Pipeline(mut dsl) = GenomeIr::parse(source)? else {
        return Err("pipeline required".into());
    };
    let mut head = dsl.stages.pop().ok_or("no head")?;
    let lambda = head.get_f64("lambda_frac", 0.02);
    if task == TaskKind::Binary {
        head = Stage::new("logit_ridge");
        head.params.insert("l2".into(), ParamValue::Float(lambda));
    }
    if task == TaskKind::Multiclass {
        head = Stage::new("multiclass_ridge");
        head.params.insert("l2".into(), ParamValue::Float(lambda));
    }
    dsl.stages.push(head);
    let GenomeIr::Pipeline(input) =
        GenomeIr::parse("(pipe X (input numeric=median missing=none categorical=native) (ridge))")?
    else {
        unreachable!()
    };
    dsl.stages.insert(0, input.stages[0].clone());
    Ok(GenomeIr::Pipeline(dsl))
}
fn metric(task: TaskKind, y: &[f64], p: &[f64]) -> Result<Value, Error> {
    if y.len() != p.len() || y.is_empty() || p.iter().any(|v| !v.is_finite()) {
        return Err("invalid prediction vector".into());
    }
    match task {
        TaskKind::Binary => {
            let eps = 1e-15;
            let loss = y
                .iter()
                .zip(p)
                .map(|(&y, &p)| {
                    if y == 0.0 {
                        -(1.0 - p.clamp(eps, 1.0 - eps)).ln()
                    } else if y == 1.0 {
                        -p.clamp(eps, 1.0 - eps).ln()
                    } else {
                        -eps.ln()
                    }
                })
                .sum::<f64>()
                / y.len() as f64;
            let accuracy = y
                .iter()
                .zip(p)
                .filter(|(y, p)| **y == if **p >= 0.5 { 1.0 } else { 0.0 })
                .count() as f64
                / y.len() as f64;
            Ok(
                json!({"selection_loss":loss,"logloss":loss,"accuracy":accuracy,"unknown_target_count":y.iter().filter(|&&v|v<0.0).count()}),
            )
        }
        TaskKind::Multiclass => {
            let f1 = feat_math::macro_f1(y, p);
            let accuracy = y.iter().zip(p).filter(|(a, b)| a == b).count() as f64 / y.len() as f64;
            Ok(
                json!({"selection_loss":1.0-f1,"macro_f1":f1,"accuracy":accuracy,"unknown_target_count":y.iter().filter(|&&v|v<0.0).count()}),
            )
        }
        _ => {
            let rmse = feat_math::rmse(y, p);
            let r2 = feat_math::r2(y, p);
            if !rmse.is_finite() {
                return Err("nonfinite score".into());
            }
            Ok(
                json!({"selection_loss":rmse,"rmse":rmse,"r2":if r2.is_finite() && y.iter().any(|v|*v!=y[0]){Some(r2)}else{None},"r2_defined":y.iter().any(|v|*v!=y[0])}),
            )
        }
    }
}
fn derive_seed(seed: u64, slot: usize, draw: usize) -> u64 {
    let b = Sha256::digest(format!("{PROTOCOL}:candidate:{seed}:{slot}:{draw}"));
    u64::from_le_bytes(b[..8].try_into().unwrap())
}
struct Measured {
    dsl: String,
    loss: f64,
    trace: usize,
}
#[cfg(test)]
fn evaluate(p: &Prepared, job: &Job, seed: u64, max_seconds: u64) -> Result<Value, Error> {
    evaluate_with_log(p, job, seed, max_seconds, None)
}
fn evaluate_with_log(
    p: &Prepared,
    job: &Job,
    seed: u64,
    max_seconds: u64,
    log: Option<&LogContext>,
) -> Result<Value, Error> {
    let start = Instant::now();
    let mut trace: Vec<Value> = Vec::new();
    let mut measured: Vec<Measured> = Vec::new();
    let mut seen = BTreeSet::new();
    let mut champion: Option<(BoundModel, String, f64, usize)> = None;
    let mut baseline: Option<(BoundModel, String)> = None;
    let mut stop = "candidate_budget";
    let mut rejected = 0usize;
    for slot in 0..CANDIDATES {
        if slot > 0 && start.elapsed().as_secs() >= max_seconds {
            stop = "seconds_before_next_fit";
            break;
        }
        let mut picked = None;
        for draw in 0..2048 {
            let childseed = derive_seed(seed, slot, draw);
            let (raw, origin, parents) = if slot == 0 {
                (
                    String::from("(pipe X (ridge lambda_frac=0.02))"),
                    "bootstrap",
                    Vec::new(),
                )
            } else if slot < 4 || measured.is_empty() {
                (
                    evolve::random_canonical_pipeline(childseed, 2),
                    "random",
                    Vec::new(),
                )
            } else {
                let mut ranking: Vec<&Measured> = measured.iter().collect();
                ranking.sort_by(|a, b| a.loss.total_cmp(&b.loss).then(a.dsl.cmp(&b.dsl)));
                let count = ranking.len().min(4);
                let a = ranking[(childseed as usize) % count];
                if slot % 3 == 0 && count > 1 {
                    let b = ranking[((childseed >> 16) as usize) % count];
                    (
                        evolve::crossover_canonical_pipelines(&a.dsl, &b.dsl, childseed, 2)
                            .map_err(|e| e.to_string())?,
                        "crossover",
                        vec![a.trace, b.trace],
                    )
                } else {
                    (
                        evolve::mutate_canonical_pipeline(&a.dsl, childseed, 2)
                            .map_err(|e| e.to_string())?,
                        "mutate",
                        vec![a.trace],
                    )
                }
            };
            if let Ok(dsl) = safe_genome(&raw) {
                if seen.insert(dsl.clone()) {
                    picked = Some((dsl, origin, parents, childseed, draw));
                    break;
                }
            }
            rejected += 1;
        }
        let Some((dsl, origin, parents, childseed, draw)) = picked else {
            stop = "bounded_grammar_exhausted";
            break;
        };
        let genome = bound_genome(&dsl, p.task.kind)?;
        let actual = genome.canonical();
        let fit_start = Instant::now();
        let fit_start_utc_us = utc_us();
        let candidate_start_relative_ns = start.elapsed().as_nanos().to_string();
        let before = memory();
        let mut row = json!({"candidate_index":slot,"generation":if slot<4 {0}else{1+(slot-4)/4},"origin":origin,"parent_candidate_indices":parents,
            "seed":childseed,"draw":draw,"search_dsl":dsl,"executed_dsl":actual,"program_sha256":hash(actual.as_bytes()),"memory_before":before});
        log_event(
            log,
            "candidate_start",
            json!({"candidate_index":slot,"origin":origin,"program_sha256":hash(actual.as_bytes())}),
        );
        // The restricted operators avoid the unsafe/unbounded kernels; no panic recovery with panic=abort.
        let fitted = feat_math::fit_table(&genome, &p.task, &p.schema, &p.fit, &p.fit_y);
        row["fit_start_utc_us"] = json!(fit_start_utc_us);
        row["fit_end_utc_us"] = json!(utc_us());
        row["candidate_start_relative_ns"] = json!(candidate_start_relative_ns);
        match fitted {
            Err(e) => {
                row["status"] = json!("fit_failed");
                row["error"] = json!(e.to_string());
            }
            Ok(model) => {
                if slot == 0 {
                    baseline = Some((model.clone(), actual.clone()));
                }
                row["selection_predict_start_utc_us"] = json!(utc_us());
                let prediction = model.predict_table(&p.selection);
                row["selection_predict_end_utc_us"] = json!(utc_us());
                match prediction
                    .map_err(|e| -> Error { e.into() })
                    .and_then(|pred| metric(p.task.kind, &p.selection_y, &pred))
                {
                    Err(e) => {
                        row["status"] = json!("selection_failed");
                        row["error"] = json!(e.to_string());
                    }
                    Ok(metrics) => {
                        let loss = metrics["selection_loss"].as_f64().ok_or("invalid loss")?;
                        row["status"] = json!("measured");
                        row["selection_metrics"] = metrics;
                        row["model_sha256"] = json!(hash(&serde_json::to_vec(&model)?));
                        measured.push(Measured {
                            dsl: dsl.clone(),
                            loss,
                            trace: slot,
                        });
                        if champion
                            .as_ref()
                            .is_none_or(|(_, d, l, _)| loss < *l || (loss == *l && actual < *d))
                        {
                            champion = Some((model, actual, loss, slot));
                        }
                    }
                }
            }
        }
        row["candidate_s"] = json!(fit_start.elapsed().as_secs_f64());
        row["candidate_end_utc_us"] = json!(utc_us());
        row["candidate_end_relative_ns"] = json!(start.elapsed().as_nanos().to_string());
        row["memory_after"] = memory();
        log_event(
            log,
            "candidate_end",
            json!({"candidate_index":slot,"status":row["status"],"model_sha256":row["model_sha256"],"candidate_s":row["candidate_s"]}),
        );
        trace.push(row);
    }
    let (model, dsl, _, index) = champion.ok_or("no GP candidate measured successfully")?;
    // Champion is frozen before audit predictions/labels are consulted. No refit or selection on audit.
    let model_bytes = serde_json::to_vec(&model)?;
    if model_bytes.len() > 3_500_000 {
        return Err("fitted artifact exceeds synchronous result bound".into());
    }
    let audit_predictions = model.predict_table(&p.audit)?;
    let audit_metrics = metric(p.task.kind, &p.audit_y, &audit_predictions)?;
    let baseline_audit = if let Some((baseline, dsl)) = baseline {
        let baseline_start = utc_us();
        let predictions = baseline.predict_table(&p.audit)?;
        let metrics = metric(p.task.kind, &p.audit_y, &predictions)?;
        let b: Vec<u8> = predictions.iter().flat_map(|v| v.to_le_bytes()).collect();
        json!({"dsl":dsl,"model_sha256":hash(&serde_json::to_vec(&baseline)?),"audit_metrics":metrics,"audit_predictions":predictions,"audit_predictions_sha256":hash(&b),
            "champion_loss_improvement":metrics["selection_loss"].as_f64().unwrap_or(0.0)-audit_metrics["selection_loss"].as_f64().unwrap_or(0.0),
            "audit_start_utc_us":baseline_start,"audit_end_utc_us":utc_us(),"audit_used_for_selection":false})
    } else {
        Value::Null
    };
    let selection_predictions = model.predict_table(&p.selection)?;
    let selection_metrics = metric(p.task.kind, &p.selection_y, &selection_predictions)?;
    let replay: BoundModel = serde_json::from_slice(&model_bytes)?;
    replay.validate_binding()?;
    let replay_predictions = replay.predict_table(&p.audit)?;
    let prediction_bytes: Vec<u8> = audit_predictions
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let replay_bytes: Vec<u8> = replay_predictions
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    if prediction_bytes != replay_bytes {
        return Err("serialized fitted-model replay mismatch".into());
    }
    Ok(
        json!({"schema":"jope.lambda-gp-job.v1","protocol":PROTOCOL,"dataset":job,
        "benchmark_valid":false,"canonical_learning_admitted":false,"forecast_valid":false,"scope":"isolated numeric corpus GP screening; not canonical evaluator qualification",
        "task_original":job.metadata.get("catalog_task").cloned().unwrap_or_else(||json!(job.task)),"task_executed":p.task,"task_alias_note":if job.metadata.get("catalog_task").is_some_and(|v|v=="multi_label") || job.task=="multi_label"{"legacy single target class code, mapped to multiclass"}else if job.task=="timeseries"{"CSV-order blocked screening; original chronological provenance unverified"}else{"none"},
        "seed":seed,"fold_policy":if job.task=="timeseries"{"csv_order_60_20_20"}else{"sha256_row_60_20_20"},"fold":p.fold,"fold_sha256":hash(&serde_json::to_vec(&p.fold)?),
        "fit_rows":p.fit.rows.len(),"selection_rows":p.selection.rows.len(),"audit_rows":p.audit.rows.len(),
        "feature_projection":{"policy":"fit_only_top64_population_variance_tie_original_index_v1","original_features":p.original_features,"selected_original_column_indices":p.columns,"schema":p.schema},
        "target_vocabulary_original_values":p.labels,"unknown_class_policy":"unseen labels encoded -1 and count as errors; fit-only vocabulary",
        "search":{"engine":"feat_math::evolve canonical mutation/crossover + actual task-bound fit_table","grammar":"max2 nonexpanding transforms; ridge-family actual task heads","candidate_limit":CANDIDATES,"candidate_attempts":trace.len(),"measured_candidates":measured.len(),"evolved_measured_candidates":trace.iter().filter(|v| v["status"]=="measured" && (v["origin"]=="mutate" || v["origin"]=="crossover")).count(),"rejected_grammar_draws":rejected,"stop_reason":stop,"soft_seconds":max_seconds,"trace":trace},
        "baseline_after_selection":baseline_audit,"champion":{"candidate_index":index,"dsl":dsl,"program_sha256":hash(dsl.as_bytes()),"model":model,"model_sha256":hash(&model_bytes),"model_json_bytes":model_bytes.len(),"selection_metrics":selection_metrics,"audit_metrics":audit_metrics,
            "audit_predictions":audit_predictions,"audit_targets_encoded":p.audit_y,"audit_predictions_sha256":hash(&prediction_bytes),"serialized_model_replay_exact":true},
        "training_s":start.elapsed().as_secs_f64(),"memory_after":memory()}),
    )
}
fn one(raw: &[u8], job: &Job, seed: u64, seconds: u64) -> Result<Value, Error> {
    one_with_log(raw, job, seed, seconds, None)
}
fn one_with_log(
    raw: &[u8],
    job: &Job,
    seed: u64,
    seconds: u64,
    log: Option<&LogContext>,
) -> Result<Value, Error> {
    let start = Instant::now();
    let prepare_start_utc_us = utc_us();
    let prepared = prepare(raw, job, seed);
    let prepare_end_utc_us = utc_us();
    let prepare_s = start.elapsed().as_secs_f64();
    let mut spans = json!({"prepare_start_utc_us":prepare_start_utc_us,"prepare_end_utc_us":prepare_end_utc_us,"prepare_start_relative_ns":"0","prepare_end_relative_ns":start.elapsed().as_nanos().to_string()});
    let evaluated = match prepared {
        Ok(p) => {
            spans["train_start_utc_us"] = json!(utc_us());
            spans["train_start_relative_ns"] = json!(start.elapsed().as_nanos().to_string());
            let result = evaluate_with_log(&p, job, seed, seconds, log);
            spans["train_end_utc_us"] = json!(utc_us());
            spans["train_end_relative_ns"] = json!(start.elapsed().as_nanos().to_string());
            result
        }
        Err(e) => Err(e),
    };
    let mut result = match evaluated {
        Ok(mut r) => {
            r["status"] = json!("trained");
            r
        }
        Err(e) => {
            json!({"status":"unsupported_or_failed","error":e.to_string(),"dataset":job,"benchmark_valid":false,"canonical_learning_admitted":false})
        }
    };
    result["timestamps"] = spans;
    result["prepare_s"] = json!(prepare_s);
    result["job_s"] = json!(start.elapsed().as_secs_f64());
    Ok(result)
}
async fn invoke(event: LambdaEvent<Request>) -> Result<Value, Error> {
    let start = Instant::now();
    let handler_start_utc_us = utc_us();
    let environment_id = environment_id();
    let req = event.payload;
    let context = LogContext {
        attempt_id: if output_key(&req.attempt_id).is_ok() {
            req.attempt_id.clone()
        } else {
            "invalid".into()
        },
        request_id: event.context.request_id.clone(),
        job_id: None,
        dataset_hash: None,
    };
    log_event(
        Some(&context),
        "handler_start",
        json!({"runtime_environment_id":environment_id,"batch_size":req.job_ids.len()}),
    );
    let output_key = match validate_output_url(&req.attempt_id, &req.output_url) {
        Ok(key) => key,
        Err(e) => {
            log_event(
                Some(&context),
                "handler_refused",
                json!({"reason":e.to_string()}),
            );
            return Err(e);
        }
    };
    let jobs = jobs()?;
    if req.job_ids.is_empty()
        || req.job_ids.len() > 5
        || req.job_ids.len() != req.download_urls.len()
        || req.max_job_seconds == 0
        || req.max_job_seconds > 600
        || req.job_ids.iter().any(|i| *i >= jobs.len())
        || req.job_ids.iter().collect::<BTreeSet<_>>().len() != req.job_ids.len()
    {
        return Err("invalid bounded batch request".into());
    }
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(45)))
        .max_redirects(0)
        .build()
        .into();
    let mut results = Vec::new();
    let mut result_bytes = 0usize;
    for (&id, url) in req.job_ids.iter().zip(&req.download_urls) {
        let job = &jobs[id];
        let job_context = LogContext {
            job_id: Some(id),
            dataset_hash: Some(job.dataset_hash.clone()),
            ..context.clone()
        };
        log_event(
            Some(&job_context),
            "job_start",
            json!({"rows":job.rows,"columns":job.columns,"csv_bytes":job.bytes}),
        );
        let download_start = Instant::now();
        let job_start_utc_us = utc_us();
        let job_start_relative_ns = start.elapsed().as_nanos().to_string();
        let mut download_end_utc_us = None;
        let mut gets = 0;
        let mut get_successes = 0;
        let mut bytes = 0;
        let result: Result<Value, Error> = if let Err(e) = admit(job) {
            Err(e)
        } else if result_bytes > 250_000
            || remaining_ms(event.context.deadline) < (req.max_job_seconds + 70) * 1000
        {
            let mut r = json!({"status":if result_bytes>250_000{"not_started_response_budget"}else{"not_started_remaining_budget"},"dataset":job,"job_id":id,"s3_get_attempts":0,"s3_get_count":0,"s3_bytes":0,"benchmark_valid":false,"canonical_learning_admitted":false});
            r["remaining_ms"] = json!(remaining_ms(event.context.deadline));
            r["timestamps"] = json!({"job_start_utc_us":job_start_utc_us,"job_end_utc_us":utc_us(),"job_start_relative_ns":job_start_relative_ns,"job_end_relative_ns":start.elapsed().as_nanos().to_string()});
            log_event(
                Some(&job_context),
                "job_end",
                json!({"status":r["status"],"candidate_attempts":0}),
            );
            results.push(r);
            continue;
        } else if job.bytes > MAX_BYTES {
            Err("input exceeds150MB protocol bound".into())
        } else {
            gets += 1;
            let downloaded = download(&agent, job, url);
            download_end_utc_us = Some(utc_us());
            downloaded.and_then(|raw| {
                get_successes += 1;
                bytes = raw.len();
                let download_s = download_start.elapsed().as_secs_f64();
                let mut result =
                    one_with_log(&raw, job, req.seed, req.max_job_seconds, Some(&job_context))?;
                result["download_s"] = json!(download_s);
                Ok(result)
            })
        };
        let mut result = match result {
            Ok(mut r) => {
                if r.get("status").is_none() {
                    r["status"] = json!("trained");
                }
                r
            }
            Err(e) => {
                json!({"status":"unsupported_or_failed","error":e.to_string(),"dataset":job,"benchmark_valid":false,"canonical_learning_admitted":false})
            }
        };
        if result.get("timestamps").is_none() {
            result["timestamps"] = json!({});
        }
        result["timestamps"]["job_start_utc_us"] = json!(job_start_utc_us);
        result["timestamps"]["job_end_utc_us"] = json!(utc_us());
        result["timestamps"]["job_start_relative_ns"] = json!(job_start_relative_ns);
        result["timestamps"]["job_end_relative_ns"] = json!(start.elapsed().as_nanos().to_string());
        if gets > 0 {
            result["timestamps"]["download_start_utc_us"] = json!(job_start_utc_us);
            result["timestamps"]["download_end_utc_us"] = json!(download_end_utc_us);
        }
        result["job_id"] = json!(id);
        result["s3_get_attempts"] = json!(gets);
        result["s3_get_count"] = json!(get_successes);
        result["s3_bytes"] = json!(bytes);
        let size = serde_json::to_vec(&result)?.len();
        if result_bytes + size > 5_000_000 {
            let spans = result["timestamps"].clone();
            result = json!({"timestamps":spans,"status":"result_oversize_retry_singleton","dataset":job,"job_id":id,"result_bytes":size,"s3_get_attempts":gets,"s3_get_count":get_successes,"s3_bytes":bytes,"benchmark_valid":false,"canonical_learning_admitted":false});
        }
        result_bytes += serde_json::to_vec(&result)?.len();
        log_event(
            Some(&job_context),
            "job_end",
            json!({"status":result["status"],"candidate_attempts":result["search"]["candidate_attempts"],"evolved_measured_candidates":result["search"]["evolved_measured_candidates"],"model_sha256":result["champion"]["model_sha256"]}),
        );
        results.push(result);
    }
    let mut response = json!({"schema":"jope.lambda-gp-batch.v1","protocol":PROTOCOL,"manifest_sha256":hash(MANIFEST.as_bytes()),"jobs":results,
        "handler_start_utc_us":handler_start_utc_us,"handler_end_utc_us":utc_us(),"handler_duration_monotonic_ns":start.elapsed().as_nanos().to_string(),"runtime_environment_id":environment_id,"handler_s":start.elapsed().as_secs_f64(),"cold":COLD.swap(false,Ordering::Relaxed),"request_id":event.context.request_id,
        "function_name":env::var("AWS_LAMBDA_FUNCTION_NAME").ok(),"function_version":env::var("AWS_LAMBDA_FUNCTION_VERSION").ok(),"architecture":env::consts::ARCH,"source_head":option_env!("JOPE_SOURCE_HEAD").unwrap_or("unrecorded"),
        "memory_mb":env::var("AWS_LAMBDA_FUNCTION_MEMORY_SIZE").ok(),"memory_after_batch":memory()});
    response["attempt_id"] = json!(req.attempt_id);
    response["handler_end_scope"] = json!("before_archive_put");
    response["archive_key"] = json!(output_key);
    let body = serde_json::to_vec(&response)?;
    log_event(
        Some(&context),
        "archive_start",
        json!({"key":output_key,"bytes":body.len(),"sha256":hash(&body)}),
    );
    let receipt = match archive_result(&req.attempt_id, &req.output_url, &body) {
        Ok(receipt) => receipt,
        Err(e) => {
            log_event(
                Some(&context),
                "archive_failed",
                json!({"key":output_key,"reason":e.to_string(),"bytes":body.len(),"sha256":hash(&body)}),
            );
            return Err(e);
        }
    };
    log_event(
        Some(&context),
        "handler_complete",
        json!({"key":output_key,"bytes":body.len(),"sha256":hash(&body),"handler_s":start.elapsed().as_secs_f64()}),
    );
    response["archive_receipt"] = receipt;
    response["response_ready_utc_us"] = json!(utc_us());
    response["handler_including_archive_s"] = json!(start.elapsed().as_secs_f64());
    Ok(response)
}
#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Error> {
    let args: Vec<String> = env::args().collect();
    if args.get(1).map(String::as_str) == Some("--local") {
        let id: usize = args
            .get(2)
            .ok_or("--local JOB_ID CSV_PATH [SEED]")?
            .parse()?;
        let raw = fs::read(args.get(3).ok_or("CSV_PATH required")?)?;
        let seed = args
            .get(4)
            .map(|s| s.parse())
            .transpose()?
            .unwrap_or(20260928);
        let jobs = jobs()?;
        let job = jobs.get(id).ok_or("unknown job")?;
        let result = one(&raw, job, seed, 600)?;
        let failed = result["status"] == "unsupported_or_failed";
        println!("{}", result);
        if failed {
            return Err("local GP job failed; detailed disposition is in JSON stdout".into());
        }
        return Ok(());
    }
    if args.get(1).map(String::as_str) == Some("--manifest-hash") {
        println!("{}", hash(MANIFEST.as_bytes()));
        return Ok(());
    }
    lambda_runtime::run(service_fn(invoke)).await
}
#[cfg(test)]
#[path = "gp_v2_tests.rs"]
mod tests;
