use feat_math::{GenomeIr, Model, Rows};
use flate2::{read::GzDecoder, write::GzEncoder, Compression};
use lambda_runtime::{service_fn, Error, LambdaEvent};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const S1_TWO_FOLD: &str = "jope.s1-two-fold/v1";

const BASE: &str = "https://veox-jopedime-learning-use2.s3.us-east-2.amazonaws.com/experiments/gp-policy-dl/v2-trainonly/";

fn hash(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn now_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_micros() as u64
}
fn is_hash(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
}
fn memory() -> Value {
    let lines: Vec<String> = fs::read_to_string("/proc/self/status")
        .unwrap_or_default()
        .lines()
        .filter(|l| {
            l.starts_with("VmPeak:")
                || l.starts_with("VmHWM:")
                || l.starts_with("VmRSS:")
                || l.starts_with("VmSwap:")
        })
        .map(str::to_string)
        .collect();
    json!(lines)
}

struct Data {
    train_x: Rows,
    train_y: Vec<f64>,
    test_x: Rows,
    test_y: Vec<f64>,
    folds: Vec<u8>,
    k: usize,
}

// JTT1: magic, n_train u32, n_test u32, p u32, k u8, n_train fold bytes,
// then row-major f64 [X..., y] for training followed by official test.
fn parse_data(raw: &[u8]) -> Result<Data, Error> {
    if raw.len() < 17 || &raw[..4] != b"JTT1" {
        return Err("invalid JTT1 header".into());
    }
    let nt = u32::from_le_bytes(raw[4..8].try_into()?) as usize;
    let nq = u32::from_le_bytes(raw[8..12].try_into()?) as usize;
    let p = u32::from_le_bytes(raw[12..16].try_into()?) as usize;
    let k = raw[16] as usize;
    if nt < 50
        || nq < 10
        || p == 0
        || p > 512
        || !(3..=16).contains(&k)
        || (nt + nq).saturating_mul(p) > 16_000_000
    {
        return Err("JTT1 shape outside bounds".into());
    }
    let expected = 17usize
        .checked_add(nt)
        .and_then(|x| x.checked_add((nt + nq).checked_mul((p + 1) * 8)?))
        .ok_or("shape overflow")?;
    if raw.len() != expected {
        return Err("JTT1 byte count mismatch".into());
    }
    let folds = raw[17..17 + nt].to_vec();
    if folds.iter().any(|f| *f as usize >= k) {
        return Err("invalid inner fold".into());
    }
    for fold in 0..k {
        if folds.iter().filter(|f| **f as usize == fold).count() < 2 {
            return Err("inner fold too small".into());
        }
    }
    let mut pos = 17 + nt;
    let mut rows = Vec::with_capacity(nt + nq);
    let mut ys = Vec::with_capacity(nt + nq);
    for row_index in 0..nt + nq {
        let mut row = Vec::with_capacity(p);
        for j in 0..=p {
            let v = f64::from_le_bytes(raw[pos..pos + 8].try_into()?);
            pos += 8;
            if row_index < nt && !v.is_finite() {
                return Err("nonfinite JTT1 value".into());
            }
            if j < p {
                row.push(v);
            } else {
                ys.push(v);
            }
        }
        rows.push(row);
    }
    Ok(Data {
        train_x: rows[..nt].to_vec(),
        train_y: ys[..nt].to_vec(),
        test_x: rows[nt..].to_vec(),
        test_y: ys[nt..].to_vec(),
        folds,
        k,
    })
}

fn metrics(y: &[f64], p: &[f64]) -> Result<Value, Error> {
    if y.is_empty() || y.len() != p.len() || p.iter().any(|v| !v.is_finite()) {
        return Err("invalid prediction".into());
    }
    let n = y.len() as f64;
    let mean = y.iter().sum::<f64>() / n;
    let sse = y.iter().zip(p).map(|(a, b)| (a - b).powi(2)).sum::<f64>();
    let tss = y.iter().map(|a| (a - mean).powi(2)).sum::<f64>();
    let mae = y.iter().zip(p).map(|(a, b)| (a - b).abs()).sum::<f64>() / n;
    let max_abs = y
        .iter()
        .zip(p)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    if !sse.is_finite() || !tss.is_finite() {
        return Err("metric overflow".into());
    }
    Ok(
        json!({"rmse":(sse/n).sqrt(),"mae":mae,"r2":if tss>0.0{Some(1.0-sse/tss)}else{None},"max_abs_error":max_abs,"sse":sse,"n":y.len()}),
    )
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Candidate {
    dsl: String,
    program_sha256: String,
    birth_seq: u64,
    parents: Vec<String>,
    operator: String,
}

fn genome(c: &Candidate) -> Result<GenomeIr, Error> {
    if c.dsl.len() > 32768
        || !is_hash(&c.program_sha256)
        || hash(c.dsl.as_bytes()) != c.program_sha256
    {
        return Err("program identity mismatch".into());
    }
    let g = GenomeIr::parse(&c.dsl)?;
    if g.canonical() != c.dsl {
        return Err("noncanonical program".into());
    }
    if let GenomeIr::Pipeline(p) = &g {
        if p.stages.len() > 9 {
            return Err("pipeline exceeds eight transforms".into());
        }
    }
    Ok(g)
}

/// Train-only input scaling (DL v2, S1, owner L6(a)): min-max of every feature column and of the target, fitted on the
/// rows a model is fitted on and only applied to the rows it scores. Inner fold j therefore depends on its inner-train
/// rows alone, and the official-test fit on the outer-train rows alone. A constant column divides by 1.
#[derive(Clone, Debug, PartialEq)]
struct Scaler {
    lo: Vec<f64>,
    span: Vec<f64>,
    y_lo: f64,
    y_span: f64,
}

fn fit_scaler(x: &Rows, y: &[f64]) -> Result<Scaler, Error> {
    if x.is_empty() || x.len() != y.len() {
        return Err("scaler needs aligned non-empty rows".into());
    }
    let p = x[0].len();
    let mut lo = vec![f64::INFINITY; p];
    let mut hi = vec![f64::NEG_INFINITY; p];
    for r in x {
        if r.len() != p {
            return Err("ragged rows".into());
        }
        for j in 0..p {
            lo[j] = lo[j].min(r[j]);
            hi[j] = hi[j].max(r[j]);
        }
    }
    let span = lo
        .iter()
        .zip(&hi)
        .map(|(l, h)| if h - l > 0.0 { h - l } else { 1.0 })
        .collect();
    let y_lo = y.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_hi = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    Ok(Scaler {
        lo,
        span,
        y_lo,
        y_span: if y_hi - y_lo > 0.0 { y_hi - y_lo } else { 1.0 },
    })
}
fn scale_rows(x: &Rows, s: &Scaler) -> Rows {
    x.iter()
        .map(|r| {
            r.iter()
                .zip(s.lo.iter().zip(&s.span))
                .map(|(v, (l, sp))| (v - l) / sp)
                .collect()
        })
        .collect()
}
fn scale_y(y: &[f64], s: &Scaler) -> Vec<f64> {
    y.iter().map(|v| (v - s.y_lo) / s.y_span).collect()
}
fn unscale_y(p: &[f64], s: &Scaler) -> Vec<f64> {
    p.iter().map(|v| v * s.y_span + s.y_lo).collect()
}
fn scaler_sha256(s: &Scaler) -> String {
    let mut b = Vec::with_capacity(8 * (2 * s.lo.len() + 2));
    for v in s.lo.iter().chain(&s.span).chain([&s.y_lo, &s.y_span]) {
        b.extend(v.to_le_bytes());
    }
    hash(&b)
}

fn one_fit(
    g: &GenomeIr,
    x: &Rows,
    y: &[f64],
    vx: &Rows,
    vy: &[f64],
    kind: &str,
    index: usize,
) -> Result<Value, Error> {
    let start = now_us();
    let tick = Instant::now();
    let s = fit_scaler(x, y)?;
    let (xs, ys, vxs) = (scale_rows(x, &s), scale_y(y, &s), scale_rows(vx, &s));
    let model = feat_math::fit(g, &xs, &ys)?;
    let fit_end = now_us();
    let train_pred = unscale_y(&model.predict(&xs)?, &s);
    let pred_s = model.predict(&vxs)?;
    let pred = unscale_y(&pred_s, &s);
    let model_bytes = serde_json::to_vec(&model)?;
    if model_bytes.len() > 32 * 1024 * 1024 {
        return Err("model exceeds 32MiB".into());
    }
    let restored: Model = serde_json::from_slice(&model_bytes)?;
    let replay = restored.predict(&vxs)?;
    if pred_s.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        != replay.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    {
        return Err("model replay mismatch".into());
    }
    let baseline = y.iter().sum::<f64>() / y.len() as f64;
    Ok(json!({
        "schema":"jope.teacher-fit/v2","kind":kind,"index":index,"train_rows":x.len(),"score_rows":vx.len(),
        "input_scaling":"minmax_fitted_on_fit_rows_v1","scaler_sha256":scaler_sha256(&s),
        "started_utc_us":start,"fit_finished_utc_us":fit_end,"finished_utc_us":now_us(),"wall_us":tick.elapsed().as_micros(),
        "metrics":metrics(vy,&pred)?,"training_metrics":metrics(y,&train_pred)?,"baseline_metrics":metrics(vy,&vec![baseline;vy.len()])?,
        "predictions":pred,"targets":vy,"model_sha256":hash(&model_bytes),"model_bytes":model_bytes.len(),"serialized_prediction_replay_exact":true
    }))
}

fn write_result(output: &Path, value: &Value) -> Result<(), Error> {
    let temporary = output.with_extension("pending");
    let mut file = File::create(&temporary)?;
    serde_json::to_writer(&mut file, value)?;
    file.flush()?;
    file.sync_all()?;
    fs::rename(&temporary, output)?;
    File::open(output.parent().ok_or("output parent missing")?)?.sync_all()?;
    Ok(())
}

fn retained_cv_result(
    output: &Path,
    child_success: bool,
    timed_out: bool,
) -> Result<Option<Value>, Error> {
    if !output.exists() {
        return Ok(None);
    }
    let mut record: Value = serde_json::from_slice(&fs::read(output)?)?;
    if record["schema"] != "jope.s1-evaluation/v1" || record["status"] != "completed" {
        return Err("invalid CV commitment".into());
    }
    if !child_success {
        record["official_test"] = json!({"status":if timed_out {"timeout"} else {"failed"},"error":"diagnostic child terminated; CV commitment retained"});
    }
    Ok(Some(record))
}

fn retained_result(
    output: &Path,
    child_success: bool,
    timed_out: bool,
    s1_two_fold: bool,
) -> Result<Option<Value>, Error> {
    if !s1_two_fold {
        return retained_cv_result(output, child_success, timed_out);
    }
    // A committed S1 CV result survives child termination without test products.
    let Some(record) = retained_cv_result(output, true, false)? else {
        return Ok(None);
    };
    if record["fold_protocol"] != S1_TWO_FOLD
        || record["inner_cv_folds"] != 2
        || record["inner_cv"].as_array().map(Vec::len) != Some(2)
        || record.get("official_test").is_some()
    {
        return Err("invalid two-fold CV commitment".into());
    }
    Ok(Some(record))
}

fn evaluate(data_path: &Path, candidate_path: &Path, output: &Path) -> Result<(), Error> {
    let d = parse_data(&fs::read(data_path)?)?;
    if d.k != 7 {
        return Err("teacher-v2 mode requires exactly seven folds".into());
    }
    let c: Candidate = serde_json::from_slice(&fs::read(candidate_path)?)?;
    let g = genome(&c)?;
    let start = now_us();
    let tick = Instant::now();
    let mut cv = Vec::with_capacity(d.k);
    for fold in 0..d.k {
        let tr: Vec<usize> = d
            .folds
            .iter()
            .enumerate()
            .filter_map(|(i, f)| ((*f as usize) != fold).then_some(i))
            .collect();
        let va: Vec<usize> = d
            .folds
            .iter()
            .enumerate()
            .filter_map(|(i, f)| ((*f as usize) == fold).then_some(i))
            .collect();
        let x = tr.iter().map(|i| d.train_x[*i].clone()).collect();
        let y: Vec<f64> = tr.iter().map(|i| d.train_y[*i]).collect();
        let vx = va.iter().map(|i| d.train_x[*i].clone()).collect();
        let vy: Vec<f64> = va.iter().map(|i| d.train_y[*i]).collect();
        cv.push(one_fit(&g, &x, &y, &vx, &vy, "inner_cv", fold)?);
    }
    let mut out = json!({"schema":"jope.s1-evaluation/v1","input_semantics":"unscaled_fit_rows_v1","program":c,"status":"completed","inner_cv":cv,
        "official_test":{"status":"pending"},"started_utc_us":start,"finished_utc_us":now_us(),"wall_us":tick.elapsed().as_micros(),"memory":memory()});
    // Commit CV atomically before any optional test work, so errors and timeout
    // during that work cannot censor CV-valid candidates.
    write_result(output, &out)?;
    out["official_test"] = match one_fit(
        &g,
        &d.train_x,
        &d.train_y,
        &d.test_x,
        &d.test_y,
        "official_test",
        0,
    ) {
        Ok(mut value) => {
            value["status"] = json!("completed");
            value
        }
        Err(error) => json!({"status":"failed","error":error.to_string()}),
    };
    out["finished_utc_us"] = json!(now_us());
    out["wall_us"] = json!(tick.elapsed().as_micros());
    write_result(output, &out)
}

fn run_child(
    c: &Candidate,
    data_path: &Path,
    dir: &Path,
    timeout_ms: u64,
    s1_two_fold: bool,
) -> Result<Value, Error> {
    let cp = dir.join("candidate.json");
    let op = dir.join("evaluation.json");
    let ep = dir.join("stderr.log");
    fs::write(&cp, serde_json::to_vec(c)?)?;
    let start = now_us();
    let tick = Instant::now();
    let mut child = Command::new(std::env::current_exe()?)
        .arg(if s1_two_fold { "--fit-s1" } else { "--fit" })
        .arg(data_path)
        .arg(&cp)
        .arg(&op)
        .stdout(Stdio::null())
        .stderr(File::create(&ep)?)
        .spawn()?;
    let mut timed_out = false;
    let status = loop {
        if let Some(s) = child.try_wait()? {
            break s;
        }
        if tick.elapsed().as_millis() >= timeout_ms as u128 {
            timed_out = true;
            child.kill()?;
            break child.wait()?;
        }
        std::thread::sleep(Duration::from_millis(5));
    };
    if let Some(record) = retained_result(&op, status.success(), timed_out, s1_two_fold)? {
        return Ok(record);
    }
    let mut failure = json!({"schema":"jope.s1-evaluation/v1","input_semantics":"unscaled_fit_rows_v1","program":c,"status":if timed_out{"timeout"}else{"failed"},
        "started_utc_us":start,"finished_utc_us":now_us(),"wall_us":tick.elapsed().as_micros(),"stderr":fs::read_to_string(ep).unwrap_or_default().chars().take(4096).collect::<String>(),"memory":memory()});
    if s1_two_fold {
        failure["fold_protocol"] = json!(S1_TWO_FOLD);
        failure["inner_cv_folds"] = json!(2);
    }
    Ok(failure)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    #[serde(default)]
    fold_protocol: Option<String>,
    schema: String,
    input_semantics: String,
    attempt_id: String,
    context_sha256: String,
    input_sha256: String,
    input_bytes: usize,
    input_url: String,
    output_url: String,
    candidate_timeout_ms: u64,
    candidate: Candidate,
}

fn allowed(url: &str, kind: &str) -> bool {
    url.starts_with(BASE)
        && !url.contains('#')
        && !url.contains("/../")
        && url[BASE.len()..].starts_with(kind)
}

fn validate_request(r: &Request) -> Result<(), Error> {
    if r.fold_protocol.as_deref().is_some_and(|p| p != S1_TWO_FOLD) {
        return Err("unsupported fold protocol".into());
    }
    if r.schema != "jope.s1-request/v1"
        || r.input_semantics != "unscaled_fit_rows_v1"
        || r.attempt_id.is_empty()
        || r.attempt_id.len() > 128
        || r.attempt_id
            .bytes()
            .any(|b| !b.is_ascii_alphanumeric() && !b"_.-".contains(&b))
        || !is_hash(&r.context_sha256)
        || !is_hash(&r.input_sha256)
        || r.input_bytes > 256 * 1024 * 1024
        || !(100..=90000).contains(&r.candidate_timeout_ms)
    {
        return Err("invalid request".into());
    }
    genome(&r.candidate)?;
    if !allowed(&r.input_url, "objects/inputs/sha256/") || !allowed(&r.output_url, "attempts/") {
        return Err("URL outside teacher prefix".into());
    }
    let expected = format!("{}attempts/{}.json.gz", BASE, r.attempt_id);
    if r.output_url.split('?').next() != Some(expected.as_str()) {
        return Err("output capability differs from attempt".into());
    }
    Ok(())
}

async fn invoke(event: LambdaEvent<Request>) -> Result<Value, Error> {
    let r = event.payload;
    let start = now_us();
    let tick = Instant::now();
    validate_request(&r)?;
    let dir = PathBuf::from(format!("/tmp/jope-teacher-{}", r.attempt_id));
    fs::create_dir(&dir)?;
    println!(
        "{}",
        json!({"event":"handler_start","attempt_id":r.attempt_id,"request_id":event.context.request_id,"utc_us":start,"birth_seq":r.candidate.birth_seq})
    );
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(30)))
        .build()
        .into();
    let mut response = agent.get(&r.input_url).call()?;
    let compressed = response
        .body_mut()
        .as_reader()
        .take(128 * 1024 * 1024 + 1)
        .read_to_vec()?;
    let mut raw = Vec::new();
    GzDecoder::new(compressed.as_slice())
        .take(256 * 1024 * 1024 + 1)
        .read_to_end(&mut raw)?;
    if raw.len() != r.input_bytes || hash(&raw) != r.input_sha256 {
        return Err("input identity mismatch".into());
    }
    let dp = dir.join("data.bin");
    fs::write(&dp, &raw)?;
    drop(raw);
    let mut record = run_child(
        &r.candidate,
        &dp,
        &dir,
        r.candidate_timeout_ms,
        r.fold_protocol.as_deref() == Some(S1_TWO_FOLD),
    )?;
    record["context_sha256"] = json!(r.context_sha256);
    let pack = json!({"schema":"jope.s1-invocation/v1","attempt_id":r.attempt_id,"aws_request_id":event.context.request_id,
        "started_utc_us":start,"finished_utc_us":now_us(),"wall_us":tick.elapsed().as_micros(),"memory":memory(),"record":record});
    let bytes = serde_json::to_vec(&pack)?;
    let mut enc = GzEncoder::new(Vec::new(), Compression::fast());
    enc.write_all(&bytes)?;
    let body = enc.finish()?;
    if body.len() > 32 * 1024 * 1024 {
        return Err("archive exceeds 32MiB".into());
    }
    agent
        .put(&r.output_url)
        .header("If-None-Match", "*")
        .header("Content-Type", "application/gzip")
        .send(&body)?;
    let out = json!({"schema":"jope.s1-invocation-receipt/v1","attempt_id":r.attempt_id,"archive_sha256":hash(&body),"archive_bytes":body.len(),"status":pack["record"]["status"]});
    println!(
        "{}",
        json!({"event":"handler_end","attempt_id":r.attempt_id,"utc_us":now_us(),"wall_us":tick.elapsed().as_micros(),"status":pack["record"]["status"]})
    );
    fs::remove_dir_all(dir)?;
    Ok(out)
}

trait ReadAll: Read {
    fn read_to_vec(mut self) -> std::io::Result<Vec<u8>>
    where
        Self: Sized,
    {
        let mut v = Vec::new();
        self.read_to_end(&mut v)?;
        Ok(v)
    }
}
impl<T: Read> ReadAll for T {}

/// S1 view: outer-test cells have no representation and are never decoded.
struct S1Data {
    x: Rows,
    y: Vec<f64>,
    folds: Vec<u8>,
}
fn parse_s1_data(raw: &[u8]) -> Result<S1Data, Error> {
    if raw.len() < 17 || &raw[..4] != b"JTT1" {
        return Err("invalid JTT1 header".into());
    }
    let nt = u32::from_le_bytes(raw[4..8].try_into()?) as usize;
    let nq = u32::from_le_bytes(raw[8..12].try_into()?) as usize;
    let p = u32::from_le_bytes(raw[12..16].try_into()?) as usize;
    if raw[16] != 2
        || nt < 50
        || nq < 10
        || p == 0
        || p > 512
        || (nt + nq).saturating_mul(p) > 16_000_000
    {
        return Err("S1 requires exactly two inner folds and an admitted shape".into());
    }
    let expected = 17usize
        .checked_add(nt)
        .and_then(|n| n.checked_add((nt + nq).checked_mul((p + 1) * 8)?))
        .ok_or("shape overflow")?;
    if raw.len() != expected {
        return Err("JTT1 byte count mismatch".into());
    }
    let folds = raw[17..17 + nt].to_vec();
    if folds.iter().any(|f| *f >= 2)
        || (0..2).any(|j| folds.iter().filter(|f| **f == j).count() < 2)
    {
        return Err("S1 header/fold membership mismatch".into());
    }
    let mut pos = 17 + nt;
    let mut x = Vec::with_capacity(nt);
    let mut y = Vec::with_capacity(nt);
    for _ in 0..nt {
        let mut row = Vec::with_capacity(p);
        for j in 0..=p {
            let v = f64::from_le_bytes(raw[pos..pos + 8].try_into()?);
            pos += 8;
            if !v.is_finite() {
                return Err("nonfinite S1 training value".into());
            }
            if j == p {
                y.push(v);
            } else {
                row.push(v);
            }
        }
        x.push(row);
    }
    Ok(S1Data { x, y, folds })
}
fn evaluate_s1(data_path: &Path, candidate_path: &Path, output: &Path) -> Result<(), Error> {
    let d = parse_s1_data(&fs::read(data_path)?)?;
    let c: Candidate = serde_json::from_slice(&fs::read(candidate_path)?)?;
    let g = genome(&c)?;
    let mut cv = Vec::with_capacity(2);
    for fold in 0..2 {
        let tr: Vec<usize> = d
            .folds
            .iter()
            .enumerate()
            .filter_map(|(i, f)| (*f != fold).then_some(i))
            .collect();
        let va: Vec<usize> = d
            .folds
            .iter()
            .enumerate()
            .filter_map(|(i, f)| (*f == fold).then_some(i))
            .collect();
        let x = tr.iter().map(|i| d.x[*i].clone()).collect();
        let y: Vec<f64> = tr.iter().map(|i| d.y[*i]).collect();
        let vx = va.iter().map(|i| d.x[*i].clone()).collect();
        let vy: Vec<f64> = va.iter().map(|i| d.y[*i]).collect();
        cv.push(one_fit(&g, &x, &y, &vx, &vy, "inner_cv", fold as usize)?);
    }
    write_result(
        output,
        &json!({"schema":"jope.s1-evaluation/v1", "fold_protocol":S1_TWO_FOLD,
        "inner_cv_folds":2, "input_semantics":"unscaled_fit_rows_v1", "program":c,
        "status":"completed", "inner_cv":cv}),
    )
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Error> {
    let a: Vec<String> = std::env::args().collect();
    if a.get(1).map(String::as_str) == Some("--fit-s1") {
        return evaluate_s1(Path::new(&a[2]), Path::new(&a[3]), Path::new(&a[4]));
    }
    if a.get(1).map(String::as_str) == Some("--fit") {
        return evaluate(Path::new(&a[2]), Path::new(&a[3]), Path::new(&a[4]));
    }
    lambda_runtime::run(service_fn(invoke)).await
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> Vec<u8> {
        let nt = 70usize;
        let nq = 14usize;
        let p = 2usize;
        let k = 7usize;
        let mut b = b"JTT1".to_vec();
        b.extend((nt as u32).to_le_bytes());
        b.extend((nq as u32).to_le_bytes());
        b.extend((p as u32).to_le_bytes());
        b.push(k as u8);
        for i in 0..nt {
            b.push((i % k) as u8)
        }
        for i in 0..nt + nq {
            for v in [i as f64, (i % 5) as f64, 2.0 * i as f64 + 1.0] {
                b.extend(v.to_le_bytes())
            }
        }
        b
    }
    #[test]
    fn format_and_metrics() {
        let d = parse_data(&fixture()).unwrap();
        assert_eq!((d.train_x.len(), d.test_x.len(), d.k), (70, 14, 7));
        assert_eq!(metrics(&[1., 2.], &[1., 2.]).unwrap()["rmse"], 0.0);
    }
    #[test]
    fn rejects_bad_fold() {
        let mut b = fixture();
        b[17] = 9;
        assert!(parse_data(&b).is_err());
    }
    const RIDGE: &str = "(pipe X (ridge))";
    const STD_RIDGE: &str = "(pipe X (standardize) (ridge))";
    fn rows(seed: u64, n: usize) -> (Rows, Vec<f64>) {
        let mut s = seed;
        let mut r = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let x: Rows = (0..n)
            .map(|_| vec![r() * 10.0 - 3.0, r() * 400.0, (r() * 4.0).floor()])
            .collect();
        let y = x
            .iter()
            .map(|v| 0.7 * v[0] - 0.01 * v[1] + 0.3 * v[2] + r() * 0.05)
            .collect();
        (x, y)
    }
    fn fit_ids(dsl: &str, x: &Rows, y: &[f64], vx: &Rows, vy: &[f64]) -> (String, String, f64) {
        let g = GenomeIr::parse(dsl).unwrap();
        let v = one_fit(&g, x, y, vx, vy, "inner_cv", 0).unwrap();
        (
            v["model_sha256"].as_str().unwrap().to_string(),
            v["scaler_sha256"].as_str().unwrap().to_string(),
            v["metrics"]["rmse"].as_f64().unwrap(),
        )
    }
    /// The fold's model and scaler are bit-identical when only its validation rows change; its score moves.
    #[test]
    fn fold_fit_ignores_its_validation_rows() {
        let (x, y) = rows(7, 60);
        let (vx, vy) = rows(9, 20);
        let mut vx2 = vx.clone();
        for r in vx2.iter_mut() {
            for v in r.iter_mut() {
                *v = *v * 1000.0 + 7.0;
            }
        }
        let vy2: Vec<f64> = vy.iter().map(|v| v * 3.0 - 50.0).collect();
        for dsl in [RIDGE, STD_RIDGE] {
            let (m1, s1, r1) = fit_ids(dsl, &x, &y, &vx, &vy);
            let (m2, s2, r2) = fit_ids(dsl, &x, &y, &vx2, &vy2);
            assert_eq!((m1, s1), (m2, s2), "{dsl}");
            assert_ne!(r1, r2, "{dsl}");
        }
    }
    /// evaluate(): inner folds and the official-test fit see only their own fit rows, end to end through a JTT1 file.
    #[test]
    fn evaluation_fits_ignore_outer_test_and_validation_rows() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../../../../target/s1-fixtures")
            .join(format!("jope-s1-eval-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let cand = |dsl: &str| {
            let c = json!({"dsl":dsl,"program_sha256":hash(dsl.as_bytes()),"birth_seq":0,"parents":[],"operator":"test"});
            serde_json::to_vec(&c).unwrap()
        };
        let run = |b: &[u8], tag: &str| -> Value {
            let dp = dir.join(format!("{tag}.jtt1"));
            let cp = dir.join(format!("{tag}.json"));
            let op = dir.join(format!("{tag}.out"));
            fs::write(&dp, b).unwrap();
            fs::write(&cp, cand(STD_RIDGE)).unwrap();
            evaluate(&dp, &cp, &op).unwrap();
            serde_json::from_slice(&fs::read(&op).unwrap()).unwrap()
        };
        let base = fixture();
        let a = run(&base, "a");
        let mut t = base.clone();
        let nt = 70usize;
        let off = 17 + nt + 8 * 3 * nt; // outer-test rows start here (p=2, so 3 values per row)
        for i in (off..t.len()).step_by(8) {
            let v = f64::from_le_bytes(t[i..i + 8].try_into().unwrap());
            t[i..i + 8].copy_from_slice(&(v * 1000.0 + 7.0).to_le_bytes());
        }
        let b = run(&t, "b");
        assert_eq!(
            a["official_test"]["model_sha256"],
            b["official_test"]["model_sha256"]
        );
        assert_eq!(
            a["official_test"]["scaler_sha256"],
            b["official_test"]["scaler_sha256"]
        );
        for j in 0..7 {
            assert_eq!(
                a["inner_cv"][j]["model_sha256"],
                b["inner_cv"][j]["model_sha256"]
            );
        }
        assert_ne!(
            a["official_test"]["metrics"]["rmse"],
            b["official_test"]["metrics"]["rmse"]
        );
        let _ = fs::remove_dir_all(&dir);
    }
    #[test]
    fn outer_diagnostic_failure_preserves_completed_cv() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../../../../target/s1-fixtures")
            .join(format!("diagnostic-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let candidate = Candidate {
            dsl: STD_RIDGE.into(),
            program_sha256: hash(STD_RIDGE.as_bytes()),
            birth_seq: 0,
            parents: vec![],
            operator: "fixture".into(),
        };
        let mut bytes = fixture();
        let target = 17 + 70 + 8 * 3 * 70 + 8 * 2;
        bytes[target..target + 8].copy_from_slice(&f64::NAN.to_le_bytes());
        let data = dir.join("data.jtt1");
        let cp = dir.join("candidate.json");
        let output = dir.join("record.json");
        fs::write(&data, &bytes).unwrap();
        fs::write(&cp, serde_json::to_vec(&candidate).unwrap()).unwrap();
        evaluate(&data, &cp, &output).unwrap();
        let actual: Value = serde_json::from_slice(&fs::read(&output).unwrap()).unwrap();
        assert_eq!(actual["status"], "completed");
        assert_eq!(actual["inner_cv"].as_array().unwrap().len(), 7);
        assert_eq!(actual["official_test"]["status"], "failed");
        let original = dir.join("original.jtt1");
        fs::write(&original, fixture()).unwrap();
        let reference = dir.join("reference.json");
        evaluate(&original, &cp, &reference).unwrap();
        let reference: Value = serde_json::from_slice(&fs::read(&reference).unwrap()).unwrap();
        for (a, b) in actual["inner_cv"]
            .as_array()
            .unwrap()
            .iter()
            .zip(reference["inner_cv"].as_array().unwrap())
        {
            assert_eq!(a["model_sha256"], b["model_sha256"]);
            assert_eq!(a["scaler_sha256"], b["scaler_sha256"]);
            assert_eq!(a["metrics"], b["metrics"]);
        }
        let timed_out = retained_cv_result(&output, false, true).unwrap().unwrap();
        assert_eq!(timed_out["inner_cv"], actual["inner_cv"]);
        assert_eq!(timed_out["official_test"]["status"], "timeout");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn scaler_is_train_rows_only_and_constant_columns_divide_by_one() {
        let x: Rows = vec![vec![1.0, 5.0], vec![3.0, 5.0]];
        let s = fit_scaler(&x, &[0.0, 2.0]).unwrap();
        assert_eq!(
            (s.lo.clone(), s.span.clone(), s.y_lo, s.y_span),
            (vec![1.0, 5.0], vec![2.0, 1.0], 0.0, 2.0)
        );
        assert_eq!(scale_rows(&vec![vec![5.0, 9.0]], &s), vec![vec![2.0, 4.0]]);
        // scored rows may leave [0, 1]
    }
    #[test]
    fn request_timeout_matches_the_ninety_second_reservation() {
        let mut r = Request {
            fold_protocol: None,
            schema: "jope.s1-request/v1".into(),
            input_semantics: "unscaled_fit_rows_v1".into(),
            attempt_id: "fixture".into(),
            context_sha256: "a".repeat(64),
            input_sha256: "b".repeat(64),
            input_bytes: 100,
            input_url: format!("{}objects/inputs/sha256/data", BASE),
            output_url: format!("{}attempts/fixture.json.gz?signature=fixture", BASE),
            candidate_timeout_ms: 90000,
            candidate: Candidate {
                dsl: STD_RIDGE.into(),
                program_sha256: hash(STD_RIDGE.as_bytes()),
                birth_seq: 0,
                parents: vec![],
                operator: "fixture".into(),
            },
        };
        validate_request(&r).unwrap();
        r.candidate_timeout_ms = 90001;
        assert!(validate_request(&r).is_err());
        r.candidate_timeout_ms = 120000;
        assert!(validate_request(&r).is_err());
        r.candidate_timeout_ms = 90000;
        r.attempt_id = "../escape".into();
        assert!(validate_request(&r).is_err());
        r.attempt_id = "fixture".into();
        r.output_url = format!("{}attempts/fixture.json.gz-extra", BASE);
        assert!(validate_request(&r).is_err());
    }

    #[test]
    fn url_scope() {
        assert!(allowed(
            &format!("{}objects/inputs/sha256/x", BASE),
            "objects/inputs/sha256/"
        ));
        assert!(!allowed("https://example.com/x", "objects/inputs/sha256/"));
    }
    fn two_fold_fixture() -> Vec<u8> {
        let mut b = fixture();
        b[16] = 2;
        for i in 0..70 {
            b[17 + i] = (i % 2) as u8;
        }
        b
    }
    #[test]
    fn s1_two_fold_view_refuses_other_counts_and_does_not_decode_test_cells() {
        let b = two_fold_fixture();
        assert!(
            parse_data(&b).is_err(),
            "original seven-fold receiver reproduces the S1 blocker"
        );
        let a = parse_s1_data(&b).unwrap();
        let mut changed = b.clone();
        changed[17 + 70 + 70 * 3 * 8..].fill(0xff); // every test cell, including targets, is now nonfinite
        let t = parse_s1_data(&changed).unwrap();
        assert_eq!((a.x, a.y, a.folds), (t.x, t.y, t.folds));
        for count in [1, 3, 7] {
            let mut bad = b.clone();
            bad[16] = count;
            assert!(parse_s1_data(&bad).is_err());
        }
        let mut bad = b.clone();
        bad[17] = 2;
        assert!(parse_s1_data(&bad).is_err());
        assert!(parse_s1_data(&b[..b.len() - 1]).is_err());
        let mut bad = b;
        bad[17 + 70..17 + 70 + 8].copy_from_slice(&f64::NAN.to_le_bytes());
        assert!(parse_s1_data(&bad).is_err());
    }
    #[test]
    fn s1_two_fold_evaluation_fits_and_scores_only_training_rows() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../../../../target/s1-fixtures")
            .join(format!("twofold-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let candidate = Candidate {
            dsl: STD_RIDGE.into(),
            program_sha256: hash(STD_RIDGE.as_bytes()),
            birth_seq: 0,
            parents: vec![],
            operator: "fixture".into(),
        };
        let cp = dir.join("candidate.json");
        fs::write(&cp, serde_json::to_vec(&candidate).unwrap()).unwrap();
        let run = |b: &[u8], tag: &str| -> Value {
            let data = dir.join(format!("{tag}.jtt1"));
            let output = dir.join(format!("{tag}.json"));
            fs::write(&data, b).unwrap();
            evaluate_s1(&data, &cp, &output).unwrap();
            serde_json::from_slice(&fs::read(output).unwrap()).unwrap()
        };
        let b = two_fold_fixture();
        let a = run(&b, "original");
        for timed_out in [false, true] {
            assert_eq!(
                a,
                retained_result(&dir.join("original.json"), false, timed_out, true)
                    .unwrap()
                    .unwrap(),
                "S1_ASSERT_TERMINATED_CV"
            );
        }
        let mut t = b.clone();
        t[17 + 70 + 70 * 3 * 8..].fill(0xff);
        let z = run(&t, "test-poison");
        assert!(a.get("official_test").is_none());
        assert_eq!(a["inner_cv_folds"], 2);
        for j in 0..2 {
            for key in [
                "model_sha256",
                "scaler_sha256",
                "predictions",
                "targets",
                "metrics",
            ] {
                assert_eq!(
                    a["inner_cv"][j][key], z["inner_cv"][j][key],
                    "S1_ASSERT_TEST_UNREAD"
                );
            }
        }
        // Authorized train-target change exercises the actual dependency.
        let mut t = b;
        for i in 0..70 {
            let off = 17 + 70 + i * 3 * 8 + 2 * 8;
            t[off..off + 8].copy_from_slice(&(i as f64 * i as f64 + 9.0).to_le_bytes());
        }
        let z = run(&t, "train-change");
        assert_ne!(
            a["inner_cv"][0]["model_sha256"],
            z["inner_cv"][0]["model_sha256"]
        );
    }
}
