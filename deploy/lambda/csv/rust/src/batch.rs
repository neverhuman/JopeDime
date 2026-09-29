use aws_credential_types::Credentials;
use aws_sigv4::http_request::{sign, PayloadChecksumKind, SignableBody, SignableRequest, SigningSettings};
use feat_math::boost2::{HistGbm2Model, HistGbm2Params};
use lambda_runtime::{service_fn, Error, LambdaEvent};
use serde_json::{json, Value};
use serde::Deserialize;
use std::{collections::BTreeMap, sync::{Arc, Mutex, OnceLock}};
use sha2::{Digest, Sha256};
use std::{env, fs, time::{Duration, Instant, SystemTime}, sync::atomic::{AtomicBool, Ordering}};

#[derive(Clone, Deserialize)]
struct Job { dataset_hash: String, name: String, key: String, sha256: String, version: String, rows: usize, columns: usize, bytes: usize }
fn jobs() -> Vec<Job> { serde_json::from_str(include_str!("../../cost/jobs.json")).expect("frozen jobs manifest") }
static COLD: AtomicBool = AtomicBool::new(true);
// At most five compile-time-pinned training inputs; cache never contains fitted models.
type CachedRows = Arc<Vec<Vec<f64>>>;
static INPUT_CACHE: OnceLock<Mutex<BTreeMap<usize, CachedRows>>> = OnceLock::new();

fn hash(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }

fn download(agent: &ureq::Agent, job: &Job) -> Result<Vec<u8>, Error> {
    let url = format!("https://veox-jopedime-datasets-use2.s3.us-east-2.amazonaws.com/{}?versionId={}", job.key, job.version);
    // Credentials are temporary values supplied by Lambda's execution role.
    let identity = Credentials::new(env::var("AWS_ACCESS_KEY_ID")?, env::var("AWS_SECRET_ACCESS_KEY")?,
        Some(env::var("AWS_SESSION_TOKEN")?), None, "lambda-role").into();
    let mut settings = SigningSettings::default();
    settings.payload_checksum_kind = PayloadChecksumKind::XAmzSha256;
    let params = aws_sigv4::sign::v4::SigningParams::builder().identity(&identity)
        .region("us-east-2").name("s3").time(SystemTime::now()).settings(settings).build()?.into();
    let request = SignableRequest::new("GET", &url, std::iter::empty(), SignableBody::Bytes(&[]))?;
    let (instructions, _) = sign(request, &params)?.into_parts();
    let mut request = agent.get(&url);
    for (name, value) in instructions.headers() { request = request.header(name, value); }
    let mut response = request.call().map_err(|e| match e {
        ureq::Error::StatusCode(code) => format!("S3 HTTP {code}; credentials and signed headers redacted"),
        _ => "S3 transport failed; credentials and signed headers redacted".to_owned(),
    })?;
    if response.headers().get("x-amz-version-id").and_then(|v| v.to_str().ok()) != Some(job.version.as_str()) {
        return Err("S3 version mismatch".into());
    }
    let bytes = response.body_mut().with_config().limit(job.bytes as u64 + 1).read_to_vec()
        .map_err(|_| "S3 body failed or exceeded admitted byte limit")?;
    Ok(bytes)
}

fn parse(raw: &[u8], job: &Job) -> Result<Vec<Vec<f64>>, Error> {
    if hash(raw) != job.sha256 || raw.len() != job.bytes { return Err("CSV hash mismatch".into()); }
    let rows: Vec<Vec<f64>> = std::str::from_utf8(raw)?.lines().map(|line| {
        line.split(',').map(str::parse::<f64>).collect::<Result<Vec<_>, _>>()
    }).collect::<Result<_, _>>()?;
    if rows.len() != job.rows || rows.iter().any(|r| r.len() != job.columns || r.iter().any(|v| !v.is_finite())) {
        return Err("CSV shape/nonfinite mismatch".into());
    }
    Ok(rows)
}

fn fold(n: usize) -> (Vec<usize>, Vec<usize>) {
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_cached_key(|i| Sha256::digest(format!("20260928:{i}").as_bytes()).to_vec());
    let mut validation: Vec<usize> = order.iter().step_by(5).copied().collect();
    validation.sort_unstable();
    let train = (0..n).filter(|i| validation.binary_search(i).is_err()).collect();
    (train, validation)
}

fn memory() -> Value {
    let status = fs::read_to_string("/proc/self/status").unwrap_or_default();
    let get = |field: &str| -> Option<u64> {
        status.lines().find(|line| line.starts_with(field))?.split_whitespace().nth(1)?.parse().ok()
    };
    json!({"rss_kib": get("VmRSS:"), "process_lifetime_hwm_kib": get("VmHWM:")})
}

fn evaluate(rows: &[Vec<f64>], job: &Job) -> Result<Value, Error> {
    let start = Instant::now();
    let (train, validation) = fold(rows.len());
    let n_feat = job.columns - 1;
    let x_train: Vec<Vec<f64>> = train.iter().map(|&i| rows[i][..n_feat].to_vec()).collect();
    let y_train: Vec<f64> = train.iter().map(|&i| rows[i][n_feat]).collect();
    let x_valid: Vec<Vec<f64>> = validation.iter().map(|&i| rows[i][..n_feat].to_vec()).collect();
    let y_valid: Vec<f64> = validation.iter().map(|&i| rows[i][n_feat]).collect();
    let cfg = HistGbm2Params { bags: 1, rounds: 500, leaves: 31, ..Default::default() };
    let prepare_s = start.elapsed().as_secs_f64();
    let before_fit = memory();
    let fit_start = Instant::now();
    let model = HistGbm2Model::fit(&x_train, &y_train, &cfg);
    let fit_s = fit_start.elapsed().as_secs_f64();
    let after_fit = memory();
    let predict_start = Instant::now();
    let predictions = model.predict(&x_valid);
    let predict_s = predict_start.elapsed().as_secs_f64();
    let mean = y_valid.iter().sum::<f64>() / y_valid.len() as f64;
    let sse: f64 = y_valid.iter().zip(&predictions).map(|(a,b)| (a-b).powi(2)).sum();
    let sst: f64 = y_valid.iter().map(|a| (a-mean).powi(2)).sum();
    let prediction_bytes: Vec<u8> = predictions.iter().flat_map(|p| p.to_le_bytes()).collect();
    let model_bytes = serde_json::to_vec(&model)?;
    let fold_json = json!({"train": train, "validation": validation});
    Ok(json!({"schema":"jope.lambda-job.v2", "algorithm":"feat-math.histgbm2",
        "algorithm_params": cfg, "dataset_hash":job.dataset_hash, "dataset_name":job.name, "csv_sha256":job.sha256,
        "fold_sha256":hash(&serde_json::to_vec(&fold_json)?), "fold_seed":20260928,
        "train_rows":train.len(),"validation_rows":validation.len(),"features":n_feat,"numeric_dtype":"f64",
        "prepare_s":prepare_s,"fit_s":fit_s,"predict_s":predict_s,
        "rmse":(sse / validation.len() as f64).sqrt(),"r2":1.0-sse/sst,
        "actual_trees":model.bags.iter().map(Vec::len).sum::<usize>(),
        "model_bytes":model_bytes.len(),"model_sha256":hash(&model_bytes),
        "predictions":predictions,"validation_ids":validation,
        "predictions_sha256":hash(&prediction_bytes),"memory_before_fit":before_fit,
        "memory_after_fit":after_fit,"memory_after_predict":memory()}))
}

async fn invoke(event: LambdaEvent<Value>) -> Result<Value, Error> {
    let start = Instant::now();
    let admitted = jobs();
    let ids: Vec<usize> = serde_json::from_value(event.payload.get("job_ids").ok_or("job_ids required")?.clone())?;
    if ids.is_empty() || ids.len() > 5 || ids.iter().any(|i| *i >= admitted.len()) || ids.iter().enumerate().any(|(i, x)| ids[..i].contains(x)) {
        return Err("batch must contain one to five distinct admitted jobs".into());
    }
    let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(20))).build().into();
    let cache_enabled = env::var("CACHE_INPUTS").as_deref() == Ok("1");
    let mut results = Vec::new();
    for id in ids {
        let job = &admitted[id];
        let job_start = Instant::now();
        let cached = if cache_enabled {
            INPUT_CACHE.get_or_init(|| Mutex::new(BTreeMap::new())).lock().map_err(|_| "cache lock poisoned")?.get(&id).cloned()
        } else { None };
        let cache_hit = cached.is_some();
        let (rows, download_s, parse_s) = if let Some(rows) = cached { (rows, 0.0, 0.0) } else {
            let raw = download(&agent, job)?;
            let download_s = job_start.elapsed().as_secs_f64();
            let parse_start = Instant::now();
            let rows = Arc::new(parse(&raw, job)?);
            let parse_s = parse_start.elapsed().as_secs_f64();
            if cache_enabled {
                INPUT_CACHE.get().expect("initialized cache").lock().map_err(|_| "cache lock poisoned")?.insert(id, rows.clone());
            }
            (rows, download_s, parse_s)
        };
        let mut result = evaluate(&rows, job)?;
        result["download_s"] = json!(download_s);
        result["parse_s"] = json!(parse_s);
        result["job_s"] = json!(job_start.elapsed().as_secs_f64());
        result["s3_bytes"] = json!(if cache_hit { 0 } else { job.bytes });
        result["s3_get_count"] = json!(if cache_hit { 0 } else { 1 });
        result["cache_hit"] = json!(cache_hit);
        results.push(result);
    }
    Ok(json!({"schema":"jope.lambda-batch.v2", "jobs":results,
        "batch_size":results.len(), "handler_s":start.elapsed().as_secs_f64(),
        "cold":COLD.swap(false, Ordering::Relaxed), "request_id":event.context.request_id,
        "memory_mb":env::var("AWS_LAMBDA_FUNCTION_MEMORY_SIZE")?.parse::<u32>()?,
        "architecture":env::consts::ARCH, "source_head":env!("JOPE_SOURCE_HEAD"),
        "memory_after_batch":memory(), "input_cache_enabled":cache_enabled}))
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Error> {
    let args: Vec<String> = env::args().collect();
    if args.get(1).map(String::as_str) == Some("--local") {
        let directory = args.get(2).ok_or("--local needs CSV directory")?;
        let mut results = Vec::new();
        for job in jobs() {
            let raw = fs::read(format!("{}/{}.csv", directory, job.dataset_hash))?;
            results.push(evaluate(&parse(&raw, &job)?, &job)?);
        }
        println!("{}", json!({"jobs":results}));
        return Ok(());
    }
    lambda_runtime::run(service_fn(invoke)).await
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fold_is_disjoint_complete_stable() {
        let (train, validation) = fold(1000);
        assert_eq!((train.len(), validation.len()), (800, 200));
        assert!(train.iter().all(|i| !validation.contains(i)));
        assert_eq!((train, validation), fold(1000));
    }
    #[test]
    fn corrupt_csv_refused() { assert!(parse(b"1,2\n", &jobs()[0]).is_err()); }
    #[test]
    fn validation_labels_do_not_change_model() {
        let rows: Vec<Vec<f64>> = (0..1000).map(|i| (0..22).map(|j| ((i*17+j*7)%101) as f64).collect()).collect();
        let first = evaluate(&rows, &jobs()[0]).unwrap();
        let mut changed = rows.clone();
        for i in fold(rows.len()).1 { changed[i][21] += 100000.0; }
        let second = evaluate(&changed, &jobs()[0]).unwrap();
        assert_eq!(first["model_sha256"], second["model_sha256"]);
        assert_eq!(first["predictions_sha256"], second["predictions_sha256"]);
        assert_ne!(first["rmse"], second["rmse"]);
    }
}
