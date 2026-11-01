//! Translate validated MCP arguments to the existing shared API transport.
use jope_api::{
    decimal::DecimalU64, ids::*, registry::Operation, Algorithm, DatasetProfile, ExportManifest,
    ExportRequest, PredictRequest, RunSnapshot, RunSpec, SystemInfo,
};
use jope_client::{Request, Transport};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, VecDeque},
    fs::File,
    io::{Cursor, Read},
    path::Path,
    sync::Mutex,
    time::Instant,
};

pub const MAX_CONTENT: usize = 2 * 1024 * 1024;
const MAX_CACHE: usize = 8 * 1024 * 1024;
#[derive(Default)]
pub struct Resources(Mutex<Cache>);
#[derive(Default)]
struct Cache {
    entries: BTreeMap<String, (String, String)>,
    order: VecDeque<String>,
    bytes: usize,
}
impl Resources {
    pub fn put(&self, uri: String, mime: &str, text: String) -> Result<(), String> {
        if text.len() > MAX_CONTENT {
            return Err("resource exceeds bounded content limit".into());
        }
        let mut cache = self.0.lock().map_err(|_| "resource cache unavailable")?;
        if let Some((_, old)) = cache.entries.remove(&uri) {
            cache.bytes -= old.len();
        }
        cache.order.retain(|u| u != &uri);
        while cache.entries.len() >= 8 || cache.bytes + text.len() > MAX_CACHE {
            let old = cache
                .order
                .pop_front()
                .ok_or("resource cache accounting failed")?;
            if let Some((_, v)) = cache.entries.remove(&old) {
                cache.bytes -= v.len();
            }
        }
        cache.bytes += text.len();
        cache.order.push_back(uri.clone());
        cache.entries.insert(uri, (mime.into(), text));
        Ok(())
    }
    pub fn get(&self, uri: &str) -> Result<Option<(String, String)>, String> {
        Ok(self
            .0
            .lock()
            .map_err(|_| "resource cache unavailable")?
            .entries
            .get(uri)
            .cloned())
    }
    pub fn list(&self) -> Result<Vec<(String, String, usize)>, String> {
        Ok(self
            .0
            .lock()
            .map_err(|_| "resource cache unavailable")?
            .entries
            .iter()
            .map(|(u, (m, t))| (u.clone(), m.clone(), t.len()))
            .collect())
    }
}

pub enum Outcome {
    Success(Value),
    Failure(Value),
}
pub fn problem(
    code: jope_api::error::ErrorCode,
    detail: impl Into<String>,
) -> jope_api::error::Problem {
    jope_api::error::Problem {
        r#type: "about:blank".into(),
        title: code.as_str().into(),
        status: code.http_status().unwrap_or(500),
        detail: detail.into(),
        instance: None,
        code,
        retriable: code
            .retriable()
            .unwrap_or(code == jope_api::error::ErrorCode::IoFailure),
        remedy: if code == jope_api::error::ErrorCode::IoFailure {
            "Check the connection; retry mutations with their original idempotency key."
        } else {
            "Supply a supported valid request."
        }
        .into(),
        field: None,
    }
}
fn invalid(text: impl Into<String>) -> jope_client::Error {
    jope_client::Error::Problem(Box::new(problem(
        jope_api::error::ErrorCode::InvalidInput,
        text,
    )))
}
fn unavailable(text: impl Into<String>) -> jope_client::Error {
    jope_client::Error::Problem(Box::new(problem(
        jope_api::error::ErrorCode::CapabilityUnavailable,
        text,
    )))
}
fn required<'a>(v: &'a Value, key: &str) -> jope_client::Result<&'a str> {
    v[key]
        .as_str()
        .filter(|v| !v.is_empty())
        .ok_or_else(|| invalid(format!("{key} must be a nonempty string")))
}
fn id<T: std::str::FromStr>(v: &Value, key: &str) -> jope_client::Result<String> {
    let text = required(v, key)?;
    text.parse::<T>()
        .map_err(|_| invalid(format!("invalid {key}")))?;
    Ok(text.into())
}
fn decimal(v: &Value, key: &str) -> jope_client::Result<Option<DecimalU64>> {
    v.get(key)
        .map(|v| serde_json::from_value(v.clone()).map_err(Into::into))
        .transpose()
}
fn json_call<T: Transport>(client: &T, r: Request) -> jope_client::Result<Value> {
    client.execute(r)?.json()
}
fn typed<T: serde::de::DeserializeOwned + serde::Serialize>(
    v: Value,
) -> jope_client::Result<Value> {
    Ok(serde_json::to_value(serde_json::from_value::<T>(v)?)?)
}
fn run(args: &Value) -> jope_client::Result<(String, Option<String>)> {
    let run = id::<RunId>(args, "run_id")?;
    let algorithm = if args.get("algorithm_id").is_some() {
        Some(id::<AlgorithmId>(args, "algorithm_id")?)
    } else {
        None
    };
    Ok((run, algorithm))
}
fn best<T: Transport>(client: &T, run: &str) -> jope_client::Result<String> {
    let snapshot: RunSnapshot = client
        .execute(Request::new(7, 1, &[("id", run)])?)?
        .json()?;
    snapshot
        .best
        .map(|best| best.algorithm_id.to_string())
        .ok_or_else(|| invalid("run has no best algorithm"))
}
fn algorithm<T: Transport>(client: &T, run: &str, alg: &str) -> jope_client::Result<Algorithm> {
    client
        .execute(Request::new(12, 1, &[("id", run), ("alg", alg)])?)?
        .json()
}

pub fn execute<T: Transport>(
    client: &T,
    resources: &Resources,
    op: &Operation,
    args: Value,
) -> Outcome {
    match execute_inner(client, resources, op, &args) {
        Ok(value) => Outcome::Success(value),
        Err(jope_client::Error::Problem(problem)) => {
            Outcome::Failure(serde_json::to_value(*problem).expect("API Problem JSON"))
        }
        Err(_) => Outcome::Failure(
            serde_json::to_value(problem(
                jope_api::error::ErrorCode::IoFailure,
                "The engine request could not be completed.",
            ))
            .expect("Problem JSON"),
        ),
    }
}
fn execute_inner<T: Transport>(
    client: &T,
    resources: &Resources,
    op: &Operation,
    args: &Value,
) -> jope_client::Result<Value> {
    let key = args.get("idempotency_key").and_then(Value::as_str);
    let request = |route, params: &[(&str, &str)]| Request::new(op.id, route, params)?.key(key);
    match op.id {
        1 => typed::<SystemInfo>(json_call(client, request(0, &[])?)?),
        2 => {
            let mut body = args.clone();
            body.as_object_mut()
                .expect("validated object")
                .remove("idempotency_key");
            if args.get("path").is_some() {
                let system: SystemInfo = client.execute(Request::new(1, 0, &[])?)?.json()?;
                if !system.local_paths {
                    return Err(unavailable("engine does not accept local paths"));
                }
            }
            typed::<DatasetProfile>(json_call(client, request(0, &[])?.json(&body)?)?)
        }
        3 => {
            if args.get("dataset_id").is_none() {
                return dataset_page(json_call(client, request(0, &[])?)?);
            }
            let dataset = id::<DatasetId>(args, "dataset_id")?;
            let mut r = request(1, &[("id", &dataset)])?;
            if let Some(columns) = args["columns"].as_array() {
                r = r.query(&[("columns", &serde_json::to_string(columns)?)]);
            }
            typed::<DatasetProfile>(json_call(client, r)?)
        }
        5 => json_call(client, request(0, &[])?.json(args)?),
        6 => {
            let mut body = args.clone();
            body.as_object_mut()
                .expect("object")
                .remove("idempotency_key");
            let body: RunSpec = serde_json::from_value(body)?;
            let value = json_call(client, request(0, &[])?.json(&body)?)?;
            let _: jope_client::RunResult = serde_json::from_value(value.clone())?;
            Ok(value)
        }
        7 => {
            if args.get("run_id").is_none() {
                return json_call(client, request(0, &[])?);
            }
            let run = id::<RunId>(args, "run_id")?;
            let mut r = request(1, &[("id", &run)])?;
            if let Some(wait) = args.get("wait_seconds") {
                r = r.query(&[("wait_seconds", &wait.to_string())]);
            }
            if let Some(after) = decimal(args, "after_event")? {
                r = r.query(&[("after_event", &after.0.to_string())]);
            }
            typed::<RunSnapshot>(json_call(client, r)?)
        }
        8 | 9 => {
            let run = id::<RunId>(args, "run_id")?;
            let mut r = request(0, &[("id", &run)])?;
            if let Some(revision) = decimal(args, "expected_revision")? {
                r = r.if_match(revision);
            }
            let mut body = args.clone();
            let map = body.as_object_mut().expect("object");
            for field in ["run_id", "idempotency_key", "expected_revision"] {
                map.remove(field);
            }
            if op.id == 8 {
                match required(args, "action")? {
                    "set_effort" if args.get("effort").is_none() => {
                        return Err(invalid("set_effort requires effort"))
                    }
                    "pause" | "continue" | "cancel"
                        if args.get("effort").is_some() || args.get("budget").is_some() =>
                    {
                        return Err(invalid("control arguments disagree with action"))
                    }
                    _ => {}
                }
            }
            typed::<RunSnapshot>(json_call(client, r.json(&body)?)?)
        }
        11 => {
            let run = id::<RunId>(args, "run_id")?;
            let after = decimal(args, "after")?.unwrap_or(DecimalU64(0));
            let limit = args["limit"].as_u64().unwrap_or(256).to_string();
            let after = after.0.to_string();
            let mut r = request(0, &[("id", &run)])?.query(&[("after", &after), ("limit", &limit)]);
            if let Some(kinds) = args["kinds"].as_array() {
                r = r.query(&[("kinds", &serde_json::to_string(kinds)?)]);
            }
            json_call(client, r)
        }
        12 => {
            let (run, alg) = run(args)?;
            let alg = match alg {
                Some(alg) => alg,
                None => best(client, &run)?,
            };
            let mut r = request(1, &[("id", &run), ("alg", &alg)])?;
            if let Some(diff) = args.get("diff_from") {
                let _: AlgorithmId = serde_json::from_value(diff.clone())?;
                r = r.query(&[("diff_from", diff.as_str().expect("string"))]);
            }
            if let Some(language) = args.get("language") {
                let _: jope_api::Language = serde_json::from_value(language.clone())?;
                r = r.query(&[("language", language.as_str().expect("string"))]);
            }
            let mut algorithm: Algorithm = serde_json::from_value(json_call(client, r)?)?;
            let max_lines = args["max_lines"].as_u64().unwrap_or(200) as usize;
            for source in &mut algorithm.source {
                if source.source.lines().count() > max_lines {
                    let language = serde_json::to_value(source.language)?
                        .as_str()
                        .expect("language")
                        .to_owned();
                    resources
                        .put(
                            format!("jope://runs/{run}/algorithms/{alg}/source/{language}"),
                            "text/plain",
                            source.source.clone(),
                        )
                        .map_err(invalid)?;
                    source.source = source
                        .source
                        .lines()
                        .take(max_lines)
                        .collect::<Vec<_>>()
                        .join("\n");
                }
            }
            Ok(serde_json::to_value(algorithm)?)
        }
        13 => {
            let output = args
                .get("out_dir")
                .map(|v| {
                    let text = v
                        .as_str()
                        .filter(|v| !v.is_empty())
                        .ok_or_else(|| invalid("out_dir must be a nonempty path"))?;
                    let path = Path::new(text);
                    jope_client::files::validate_export_destination(path)
                        .map_err(|e| invalid(e.to_string()))?;
                    Ok::<_, jope_client::Error>(path)
                })
                .transpose()?;
            let (run, alg) = run(args)?;
            let alg = match alg {
                Some(alg) => alg,
                None => best(client, &run)?,
            };
            let body: ExportRequest = serde_json::from_value(
                json!({"algorithm_id":alg,"kind":args["kind"],"languages":args["languages"],"require_all_languages":true,"refit":"fit_holdout"}),
            )?;
            let manifest: ExportManifest = serde_json::from_value(json_call(
                client,
                request(0, &[("id", &run)])?.json(&body)?,
            )?)?;
            if let Some(output) = output {
                jope_client::files::download_export_artifact(
                    client,
                    &manifest.artifact_id,
                    output,
                    false,
                )
                .map_err(|error| invalid(error.to_string()))?;
            }
            resources
                .put(
                    format!("jope://artifacts/{}/manifest", manifest.artifact_id),
                    "application/json",
                    serde_json::to_string(&manifest)?,
                )
                .map_err(invalid)?;
            Ok(serde_json::to_value(manifest)?)
        }
        15 => predict(client, resources, args, key),
        _ => Err(invalid("operation is not an MCP tool")),
    }
}
fn dataset_page(value: Value) -> jope_client::Result<Value> {
    let mut page = value
        .as_object()
        .cloned()
        .ok_or_else(|| invalid("dataset page must be an object"))?;
    if page
        .keys()
        .any(|name| !matches!(name.as_str(), "datasets" | "items" | "next_after"))
    {
        return Err(invalid(
            "unmaterialized dataset page contains unknown fields",
        ));
    }
    let field = if page.contains_key("datasets") && !page.contains_key("items") {
        "datasets"
    } else if page.contains_key("items") && !page.contains_key("datasets") {
        "items"
    } else {
        return Err(invalid("unmaterialized dataset page shape"));
    };
    let profiles: Vec<DatasetProfile> = serde_json::from_value(page[field].clone())?;
    page.insert(field.into(), serde_json::to_value(profiles)?);
    Ok(Value::Object(page))
}
fn predict<T: Transport>(
    client: &T,
    resources: &Resources,
    args: &Value,
    key: Option<&str>,
) -> jope_client::Result<Value> {
    let model = args["model"]
        .as_object()
        .ok_or_else(|| invalid("model must be an object"))?;
    let artifact = if model.len() == 1 && model.contains_key("artifact_id") {
        id::<ArtifactId>(&args["model"], "artifact_id")?
    } else if model.len() == 1 && model.contains_key("path") {
        let path = required(&args["model"], "path")?;
        path.parse::<ArtifactId>()
            .map_err(|_| invalid("model.path must be an explicit artifact ID"))?;
        path.to_owned()
    } else {
        if model
            .keys()
            .any(|k| !matches!(k.as_str(), "run_id" | "algorithm_id"))
        {
            return Err(invalid("model reference has unknown fields"));
        }
        let (run, alg) = run(&args["model"])?;
        let alg = match alg {
            Some(alg) => alg,
            None => best(client, &run)?,
        };
        algorithm(client, &run, &alg)?
            .model_artifact
            .ok_or_else(|| invalid("algorithm has no model artifact"))?
            .to_string()
    };
    let input = args["input"]
        .as_object()
        .ok_or_else(|| invalid("input must be an object"))?;
    let dataset = if input.len() == 1 && input.contains_key("dataset_id") {
        id::<DatasetId>(&args["input"], "dataset_id")?
    } else if input.len() == 1 && input.contains_key("path") {
        let path_text = required(&args["input"], "path")?;
        let path = Path::new(path_text);
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .filter(|name| !name.is_empty())
            .ok_or_else(|| invalid("input path must name a UTF-8 file"))?;
        let file = File::open(path)?;
        let key = key.ok_or_else(|| invalid("missing idempotency key"))?;
        jope_client::files::upload_csv(client, name, file, key)?
            .json::<DatasetProfile>()?
            .dataset_id
            .to_string()
    } else {
        return Err(invalid("input must name one dataset_id or local path"));
    };
    let body: PredictRequest =
        serde_json::from_value(json!({"model_artifact":artifact,"dataset":dataset}))?;
    let started = Instant::now();
    let response = client.model_predict(
        &body,
        key.ok_or_else(|| invalid("missing idempotency key"))?,
    )?;
    let mut bytes = Vec::new();
    response
        .body
        .take((MAX_CONTENT + 1) as u64)
        .read_to_end(&mut bytes)?;
    if bytes.len() > MAX_CONTENT {
        return Err(invalid(
            "prediction response exceeds bounded resource limit",
        ));
    }
    let table = jope_runtime::table::parse_csv(&bytes, &Default::default())
        .map_err(|_| invalid("invalid prediction CSV"))?;
    let hash = jope_runtime::sha256_hex(&bytes);
    if let Some(out_path) = args.get("out_path") {
        let out_path = out_path
            .as_str()
            .filter(|path| !path.is_empty())
            .ok_or_else(|| invalid("out_path must be a nonempty string"))?;
        jope_client::files::write_csv_atomic(Path::new(out_path), Cursor::new(&bytes))
            .map_err(|error| invalid(error.to_string()))?;
        return Ok(
            json!({"rows":table.rows.len().to_string(),"sha256":hash,"wall_seconds":started.elapsed().as_secs_f64(),"out_path":out_path}),
        );
    }
    let uri = format!("jope://predictions/{hash}");
    resources
        .put(
            uri.clone(),
            "text/csv",
            String::from_utf8(bytes).map_err(|_| invalid("prediction CSV is not UTF-8"))?,
        )
        .map_err(invalid)?;
    Ok(
        json!({"rows":table.rows.len().to_string(),"sha256":hash,"wall_seconds":started.elapsed().as_secs_f64(),"resource":uri}),
    )
}

pub fn read_resource<T: Transport>(
    client: &T,
    resources: &Resources,
    uri: &str,
) -> Result<(String, String), String> {
    if let Some(value) = resources.get(uri)? {
        return Ok(value);
    }
    let path = uri
        .strip_prefix("jope://")
        .ok_or("unsupported resource scheme")?;
    let pieces: Vec<_> = path.split('/').collect();
    let v = match pieces.as_slice() {
        ["runs", run] => {
            let _: RunId = run.parse().map_err(|_| "invalid run resource")?;
            client
                .execute(Request::new(7, 1, &[("id", run)]).map_err(|e| e.to_string())?)
                .and_then(|r| r.json::<RunSnapshot>())
                .and_then(|v| serde_json::to_value(v).map_err(Into::into))
        }
        ["datasets", dataset, "profile"] => {
            let _: DatasetId = dataset.parse().map_err(|_| "invalid dataset resource")?;
            client
                .execute(Request::new(3, 1, &[("id", dataset)]).map_err(|e| e.to_string())?)
                .and_then(|r| r.json::<DatasetProfile>())
                .and_then(|v| serde_json::to_value(v).map_err(Into::into))
        }
        ["runs", run, "algorithms", alg, "source", language] => {
            let _: RunId = run.parse().map_err(|_| "invalid run resource")?;
            let _: AlgorithmId = alg.parse().map_err(|_| "invalid algorithm resource")?;
            let language: jope_api::Language =
                serde_json::from_value(json!(language)).map_err(|_| "invalid source language")?;
            let algorithm =
                algorithm(client, run, alg).map_err(|_| "source resource unavailable")?;
            let source = algorithm
                .source
                .into_iter()
                .find(|s| s.language == language)
                .ok_or("source language unavailable")?
                .source;
            if source.len() > MAX_CONTENT {
                return Err("source resource exceeds limit".into());
            }
            return Ok(("text/plain".into(), source));
        }
        ["artifacts", _, "manifest"] => {
            return Err(
                "manifest unavailable: artifact manifest is not cached in this process".into(),
            )
        }
        _ => return Err("resource not found".into()),
    }
    .map_err(|_| "engine resource unavailable")?;
    let text = serde_json::to_string(&v).map_err(|_| "resource serialization failed")?;
    if text.len() > MAX_CONTENT {
        return Err("resource exceeds limit".into());
    }
    Ok(("application/json".into(), text))
}

#[cfg(test)]
mod export_destination_tests {
    use super::*;
    struct NoCalls;
    impl Transport for NoCalls {
        fn execute(&self, _: Request) -> jope_client::Result<jope_client::transport::Response> {
            panic!("invalid destination reached the engine")
        }
    }
    #[test]
    fn empty_export_destination_is_rejected_before_any_transport_call() {
        let op = jope_api::registry::OPERATIONS
            .iter()
            .find(|v| v.id == 13)
            .unwrap();
        let args = json!({"run_id":RunId::new(), "algorithm_id":"alg_0001", "kind":"bundle", "languages":["rust"], "out_dir":"", "idempotency_key":"retry"});
        let Outcome::Failure(problem) = execute(&NoCalls, &Resources::default(), op, args) else {
            panic!("empty destination succeeded")
        };
        assert_eq!(problem["code"], "INVALID_INPUT");
    }
}
