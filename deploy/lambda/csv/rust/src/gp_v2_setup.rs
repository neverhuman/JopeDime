use super::*;

pub(super) fn default_bucket() -> String {
    DATASET_BUCKET.to_owned()
}
pub(super) const MAX_BYTES: usize = 150_000_000;
pub(super) const MAX_ROWS: usize = 50_000;
pub(super) const MAX_COLS: usize = 21_000;
pub(super) const WIDTH: usize = 64;
pub(super) const CANDIDATES: usize = 12;
pub(super) static COLD: AtomicBool = AtomicBool::new(true);
pub(super) static ENVIRONMENT: std::sync::OnceLock<String> = std::sync::OnceLock::new();
pub(super) fn utc_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default
        ()
        .as_micros() as u64
}
pub(super) fn environment_id() -> &'static str {
    ENVIRONMENT.get_or_init(|| {
        hash(
            format!(
                "{}:{}:{}",
                env::var("AWS_LAMBDA_LOG_STREAM_NAME").unwrap_or_default
                (),
                std::process::id(),
                utc_us()
            )
            .as_bytes(),
        )
    })
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct Job {
    #[serde(default = "default_bucket")]
    pub(super) bucket: String,
    pub(super) dataset_hash: String,
    pub(super) name: String,
    pub(super) key: String,
    pub(super) sha256: String,
    pub(super) version: String,
    pub(super) rows: usize,
    pub(super) columns: usize,
    pub(super) bytes: usize,
    pub(super) task: String,
    #[serde(default)]
    pub(super) partition: String,
    #[serde(flatten)]
    pub(super) metadata: BTreeMap<String, Value>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Request {
    pub(super) attempt_id: String,
    pub(super) output_url: String,
    pub(super) job_ids: Vec<usize>,
    pub(super) download_urls: Vec<String>,
    pub(super) seed: u64,
    #[serde(default = "default_seconds")]
    pub(super) max_job_seconds: u64,
}

pub(super) const OUTPUT_BUCKET: &str = "veox-jopedime-learning-use2";
pub(super) const OUTPUT_PREFIX: &str =
    "experiments/lambda-gp/v1/campaigns/gp-corpus-20260928-v1/raw-attempts/";
#[derive(Clone)]
pub(super) struct LogContext {
    pub(super) attempt_id: String,
    pub(super) request_id: String,
    pub(super) job_id: Option<usize>,
    pub(super) dataset_hash: Option<String>,
}
pub(super) fn log_event(context: Option<&LogContext>, event: &str, details: Value) {
    if let Some(c) = context {
        let mut row = json!({"schema":"jope.lambda-gp-log.v1","event":event,"utc_us":utc_us(),"attempt_id":c.attempt_id,"request_id":c.request_id,"job_id":c.job_id,"dataset_hash":c.dataset_hash});
        if let Some(fields) = details.as_object() {
            for (key, value) in fields {
                row[key] = value.clone();
            }
        }
        println!("{}", row);
    }
}
pub(super) fn output_key(attempt_id: &str) -> Result<String, Error> {
    if attempt_id.len() != 32
        || !attempt_id
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err("attempt_id must be32lowercasehexcharacters".into());
    }
    Ok(format!("{OUTPUT_PREFIX}{attempt_id}.json"))
}
pub(super) fn validate_output_url(attempt_id: &str, url: &str) -> Result<String, Error> {
    let key = output_key(attempt_id)?;
    if url.len() > 8192 || url.contains('#') {
        return Err("invalid archive capability URL".into());
    }
    let (path, query) = url
        .split_once('?')
        .ok_or("archive capability query missing")?;
    if path != format!("https://{OUTPUT_BUCKET}.s3.us-east-2.amazonaws.com/{key}") {
        return Err("archive URL outside exact attempt object".into());
    }
    let fields = query
        .split('&')
        .map(|v| v.split_once('=').ok_or("invalid archive query"))
        .collect::<Result<Vec<_>, _>>()?;
    let value = |key: &str| -> Option<&str> {
        let mut found = fields.iter().filter(|(k, _)| *k == key);
        let first = found.next().map(|(_, v)| *v)?;
        if found.next().is_some() {
            None
        } else {
            Some(first)
        }
    };
    let expires = value("X-Amz-Expires")
        .and_then(|s| s.parse::<u64>().ok())
        .ok_or("archive expiry missing")?;
    if expires == 0 || expires > 900 {
        return Err("archive expiry outside15minute bound".into());
    }
    if value("X-Amz-Algorithm") != Some("AWS4-HMAC-SHA256") {
        return Err("archive requiresSigV4".into());
    }
    if value("X-Amz-Credential").is_none_or(str::is_empty) {
        return Err("archive credential binding missing".into());
    }
    let date = value("X-Amz-Date").ok_or("archive date missing")?;
    if date.len() != 16
        || date.as_bytes()[8] != b'T'
        || date.as_bytes()[15] != b'Z'
        || !date
            .bytes()
            .enumerate()
            .all(|(i, b)| i == 8 || i == 15 || b.is_ascii_digit())
    {
        return Err("archive SigV4 date malformed".into());
    }
    let signature = value("X-Amz-Signature").ok_or("archive signature missing")?;
    if signature.len() != 64 || !signature.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("archive signature invalid".into());
    }
    let headers = value("X-Amz-SignedHeaders")
        .ok_or("archive signedheaders missing")?
        .replace("%3B", ";")
        .replace("%3b", ";");
    for header in [
        "content-type",
        "host",
        "if-none-match",
        "x-amz-server-side-encryption",
    ] {
        if !headers.split(';').any(|v| v == header) {
            return Err(
                "archive must bind contenttype conditionalwrite andencryption headers".into(),
            );
        }
    }
    Ok(key)
}
pub(super) fn archive_result(attempt_id: &str, url: &str, body: &[u8]) -> Result<Value, Error> {
    let key = validate_output_url(attempt_id, url)?;
    if body.len() > 5_500_000 {
        return Err("archive body exceeds5.5MB bound".into());
    }
    let start = Instant::now();
    let begin = utc_us();
    // A fresh agent performs exactly one conditional PUT; there is no application retry.
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(15)))
        .max_redirects(0)
        .build()
        .into();
    let response=agent.put(url).header("content-type","application/json").header("if-none-match","*").header("x-amz-server-side-encryption","AES256").send(body)
        .map_err(|_|"archive PUT failed or object already exists; capability URL redacted; recover deterministic attempt key")?;
    let header = |key: &str| {
        response
            .headers()
            .get(key)
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned)
    };
    if header("x-amz-server-side-encryption").as_deref() != Some("AES256") {
        return Err(
            "archive encryption acknowledgement mismatch; inspect deterministic attempt key".into(),
        );
    }
    let version = header("x-amz-version-id")
        .filter(|v| !v.is_empty() && v != "null")
        .ok_or("archive version acknowledgement missing; recover deterministic attempt key")?;
    Ok(
        json!({"bucket":OUTPUT_BUCKET,"key":key,"bytes":body.len(),"sha256":hash(body),"etag":header("etag"),"version":version,"put_start_utc_us":begin,"put_end_utc_us":utc_us(),"put_s":start.elapsed().as_secs_f64(),"conditional_if_none_match":"*","server_side_encryption":"AES256"}),
    )
}
