//! Isolated, bounded Lambda GP screening. Never canonical benchmark/learning admission.
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
const MANIFEST: &str = include_str!(env!("JOPE_GP_MANIFEST_PATH"));
const PROTOCOL: &str = "jope.lambda-gp-screen.v1";
const DATASET_BUCKET: &str = "veox-jopedime-datasets-use2";
const LEARNING_BUCKET: &str = "veox-jopedime-learning-use2";
const CURATED_PREFIX: &str = "experiments/lambda-gp/v1/campaigns/gp-corpus-20260928-v1/";
fn default_bucket() -> String {
    DATASET_BUCKET.to_owned()
}
const MAX_BYTES: usize = 150_000_000;
const MAX_ROWS: usize = 50_000;
const MAX_COLS: usize = 21_000;
const WIDTH: usize = 64;
const CANDIDATES: usize = 12;
static COLD: AtomicBool = AtomicBool::new(true);
static ENVIRONMENT: std::sync::OnceLock<String> = std::sync::OnceLock::new();
fn utc_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}
fn environment_id() -> &'static str {
    ENVIRONMENT.get_or_init(|| {
        hash(
            format!(
                "{}:{}:{}",
                env::var("AWS_LAMBDA_LOG_STREAM_NAME").unwrap_or_default(),
                std::process::id(),
                utc_us()
            )
            .as_bytes(),
        )
    })
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Job {
    #[serde(default = "default_bucket")]
    bucket: String,
    dataset_hash: String,
    name: String,
    key: String,
    sha256: String,
    version: String,
    rows: usize,
    columns: usize,
    bytes: usize,
    task: String,
    #[serde(default)]
    partition: String,
    #[serde(flatten)]
    metadata: BTreeMap<String, Value>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    job_ids: Vec<usize>,
    download_urls: Vec<String>,
    seed: u64,
    #[serde(default = "default_seconds")]
    max_job_seconds: u64,
}
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
fn evaluate(p: &Prepared, job: &Job, seed: u64, max_seconds: u64) -> Result<Value, Error> {
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
            let result = evaluate(&p, job, seed, seconds);
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
            || remaining_ms(event.context.deadline) < (req.max_job_seconds + 50) * 1000
        {
            let mut r = json!({"status":if result_bytes>250_000{"not_started_response_budget"}else{"not_started_remaining_budget"},"dataset":job,"job_id":id,"s3_get_attempts":0,"s3_get_count":0,"s3_bytes":0,"benchmark_valid":false,"canonical_learning_admitted":false});
            r["remaining_ms"] = json!(remaining_ms(event.context.deadline));
            r["timestamps"] = json!({"job_start_utc_us":job_start_utc_us,"job_end_utc_us":utc_us(),"job_start_relative_ns":job_start_relative_ns,"job_end_relative_ns":start.elapsed().as_nanos().to_string()});
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
                let mut result = one(&raw, job, req.seed, req.max_job_seconds)?;
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
        results.push(result);
    }
    Ok(
        json!({"schema":"jope.lambda-gp-batch.v1","protocol":PROTOCOL,"manifest_sha256":hash(MANIFEST.as_bytes()),"jobs":results,
        "handler_start_utc_us":handler_start_utc_us,"handler_end_utc_us":utc_us(),"handler_duration_monotonic_ns":start.elapsed().as_nanos().to_string(),"runtime_environment_id":environment_id,"handler_s":start.elapsed().as_secs_f64(),"cold":COLD.swap(false,Ordering::Relaxed),"request_id":event.context.request_id,
        "architecture":env::consts::ARCH,"source_head":option_env!("JOPE_SOURCE_HEAD").unwrap_or("unrecorded"),
        "memory_mb":env::var("AWS_LAMBDA_FUNCTION_MEMORY_SIZE").ok(),"memory_after_batch":memory()}),
    )
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
mod tests {
    use super::*;
    fn fixture(task: &str) -> (Vec<u8>, Job) {
        let raw = (0..100)
            .map(|i| {
                format!(
                    "{},{},{},{}\n",
                    i as f64 / 100.0,
                    (i * 17 % 101) as f64 / 100.0,
                    (i * 31 % 97) as f64 / 100.0,
                    if task == "regression" {
                        i as f64 * 0.7
                    } else if task == "binary" {
                        (i % 2) as f64
                    } else {
                        (i % 3) as f64 / 2.0
                    }
                )
            })
            .collect::<String>()
            .into_bytes();
        let job = Job {
            bucket: default_bucket(),
            dataset_hash: "fixture".into(),
            name: "fixture".into(),
            key: "blobs/sha256/fixture".into(),
            sha256: hash(&raw),
            version: "v1".into(),
            rows: 100,
            columns: 4,
            bytes: raw.len(),
            task: task.into(),
            partition: "experiment".into(),
            metadata: BTreeMap::new(),
        };
        (raw, job)
    }
    #[test]
    fn serialized_float_state_preserves_ieee_bits() {
        let mut bits = 0x123456789abcdef0u64;
        for _ in 0..10000 {
            bits = bits
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let x = f64::from_bits(bits);
            if x.is_finite() {
                let encoded = serde_json::to_vec(&x).unwrap();
                let decoded: f64 = serde_json::from_slice(&encoded).unwrap();
                assert_eq!(
                    x.to_bits(),
                    decoded.to_bits(),
                    "{}",
                    String::from_utf8(encoded).unwrap()
                );
            }
        }
    }
    #[test]
    fn folds_disjoint_complete() {
        for n in 5..500 {
            let f = fold(n, "x", 42, false);
            let union: BTreeSet<usize> = f
                .fit
                .iter()
                .chain(&f.selection)
                .chain(&f.audit)
                .copied()
                .collect();
            assert_eq!(union.len(), n);
            assert_eq!(f.fit.len() + f.selection.len() + f.audit.len(), n);
            assert!(!f.audit.is_empty());
        }
    }
    #[test]
    fn curated_bucket_requires_exact_namespace() {
        let (_, mut job) = fixture("regression");
        assert!(admit(&job).is_ok());
        job.bucket = LEARNING_BUCKET.into();
        assert!(admit(&job).is_err());
        job.key = format!("{}curated-inputs/example.csv", CURATED_PREFIX);
        assert!(admit(&job).is_ok());
        job.bucket = "unrelated-bucket".into();
        assert!(admit(&job).is_err());
    }
    #[test]
    fn sealed_metadata_refused() {
        let (_, mut job) = fixture("regression");
        job.metadata.insert("sealed".into(), json!(true));
        assert!(admit(&job).is_err());
        job.metadata.remove("sealed");
        job.metadata.insert(
            "historical_partitions".into(),
            json!(["meta_train", "meta_test"]),
        );
        assert!(admit(&job).is_err());
    }
    #[test]
    fn malformed_and_nonfinite_cells_refused() {
        for s in ["NaNxxx", "inf", "1\"", "\"1", "\"1\"2"] {
            assert!(number(s).is_err());
        }
        assert_eq!(number("\"1.5\"").unwrap(), Some(1.5));
    }
    #[test]
    fn failure_keeps_stage_timestamps() {
        let (raw, mut job) = fixture("regression");
        job.sha256 = "0".repeat(64);
        let value = one(&raw, &job, 42, 25).unwrap();
        assert_eq!(value["status"], "unsupported_or_failed");
        let t = &value["timestamps"];
        assert!(t["prepare_start_utc_us"].as_u64().is_some());
        assert!(t["prepare_end_utc_us"].as_u64().is_some());
        assert!(t.get("train_start_utc_us").is_none());
    }
    #[test]
    fn corrupted_input_rejected() {
        let (raw, mut job) = fixture("regression");
        job.sha256 = "0".repeat(64);
        assert!(prepare(&raw, &job, 42).is_err());
    }
    #[test]
    fn costly_grammar_rejected() {
        assert!(safe_genome("(pipe X (gpr))").is_err());
        assert!(safe_genome("(pipe X (cheb degree=4) (ridge))").is_err());
        assert!(safe_genome("(pipe X (rank append=true) (ridge))").is_err());
    }
    #[test]
    fn classification_and_regression_replay_and_audit_isolation() {
        for task in ["regression", "binary", "multi_label"] {
            let (raw, job) = fixture(task);
            let mut p = prepare(&raw, &job, 42).unwrap();
            let a = evaluate(&p, &job, 42, 600).unwrap();
            assert_eq!(a["search"]["candidate_attempts"], 12);
            assert_eq!(a["champion"]["serialized_model_replay_exact"], true);
            assert!(a["search"]["trace"]
                .as_array()
                .unwrap()
                .iter()
                .any(|t| t["origin"] == "mutate"));
            for y in &mut p.audit_y {
                *y += 10000.0;
            }
            let b = evaluate(&p, &job, 42, 600).unwrap();
            assert_eq!(a["champion"]["model_sha256"], b["champion"]["model_sha256"]);
            assert_eq!(a["champion"]["dsl"], b["champion"]["dsl"]);
            assert_eq!(
                a["champion"]["audit_predictions_sha256"],
                b["champion"]["audit_predictions_sha256"]
            );
            assert_eq!(
                a["baseline_after_selection"]["audit_predictions_sha256"],
                b["baseline_after_selection"]["audit_predictions_sha256"]
            );
            let baseline_predictions: Vec<f64> =
                serde_json::from_value(a["baseline_after_selection"]["audit_predictions"].clone())
                    .unwrap();
            assert_eq!(
                hash(
                    &baseline_predictions
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>()
                ),
                a["baseline_after_selection"]["audit_predictions_sha256"]
            );
            assert_eq!(
                a["baseline_after_selection"]["audit_used_for_selection"],
                false
            );
            assert_ne!(
                a["champion"]["audit_metrics"],
                b["champion"]["audit_metrics"]
            );
        }
    }
    #[test]
    fn projection_and_vocabulary_ignore_audit_data() {
        for task in ["regression", "binary", "multi_label"] {
            let (raw, mut job) = fixture(task);
            let p = prepare(&raw, &job, 42).unwrap();
            let poisoned = std::str::from_utf8(&raw)
                .unwrap()
                .lines()
                .enumerate()
                .map(|(i, line)| {
                    if p.fold.audit.contains(&i) {
                        let mut row = line.split(',').map(str::to_owned).collect::<Vec<_>>();
                        row[0] = "999999999".into();
                        row[3] = "987654321".into();
                        row.join(",") + "\n"
                    } else {
                        line.to_owned() + "\n"
                    }
                })
                .collect::<String>()
                .into_bytes();
            job.sha256 = hash(&poisoned);
            job.bytes = poisoned.len();
            let q = prepare(&poisoned, &job, 42).unwrap();
            assert_eq!(p.columns, q.columns);
            assert_eq!(p.labels, q.labels);
            assert_eq!(p.fit.rows, q.fit.rows);
            assert_eq!(p.fit_y, q.fit_y);
            assert_eq!(p.selection.rows, q.selection.rows);
            assert_eq!(p.selection_y, q.selection_y);
        }
        let make = |poison: bool| {
            let f = fold(100, "wide", 42, false);
            (0..100)
                .map(|i| {
                    let mut row = (0..70)
                        .map(|j| {
                            if poison && f.audit.contains(&i) && j == 69 {
                                "10000000".into()
                            } else {
                                format!("{}", (i * (j + 1)) % 101)
                            }
                        })
                        .collect::<Vec<_>>();
                    row.push(i.to_string());
                    row.join(",") + "\n"
                })
                .collect::<String>()
                .into_bytes()
        };
        let raw = make(false);
        let mut job = Job {
            bucket: default_bucket(),
            dataset_hash: "wide".into(),
            name: "wide".into(),
            key: "blobs/sha256/x".into(),
            sha256: hash(&raw),
            version: "v".into(),
            rows: 100,
            columns: 71,
            bytes: raw.len(),
            task: "regression".into(),
            partition: "".into(),
            metadata: BTreeMap::new(),
        };
        let p = prepare(&raw, &job, 42).unwrap();
        let raw = make(true);
        job.sha256 = hash(&raw);
        job.bytes = raw.len();
        let q = prepare(&raw, &job, 42).unwrap();
        assert_eq!(p.columns.len(), 64);
        assert_eq!(p.columns, q.columns);
        assert_eq!(p.fit.rows, q.fit.rows);
        assert_eq!(
            p.fit.rows.len() + p.selection.rows.len() + p.audit.rows.len(),
            100
        );
    }
}
