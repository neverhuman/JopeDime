//! Read-only GPU timing-study reporting for v2 and v3 journals. Missing, corrupt and unsupported
//! are explicit: an absent attempt is unexecuted, never zero latency.
//!
//! Receipts are judged exactly as the retired Python reporter judged them, including its
//! equality and truthiness rules. JSON keeps its file member order and the summary is written
//! with Python `json.dumps(indent=2)` spelling (ASCII escapes, `repr` floats), so the report
//! bytes do not depend on the implementation language. A receipt member whose JSON type the
//! checks cannot inspect fails the report, as it did before, rather than being guessed at.

use anyhow::{bail, ensure, Context, Result};
use serde::de::{Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use sha2::{Digest, Sha256};
use std::{
    cmp::Ordering,
    fmt::{self, Write as _},
    fs,
    path::{Path, PathBuf},
};

const SUPPORTED: [&str; 2] = ["jope.inference-benchmark/v3", "jope.inference-benchmark/v2"];
/// Direct-request census of the v2 study when a plan predates an explicit census.
const DEFAULT_CENSUS: [(&str, u64); 4] = [("warm", 10), ("cold", 1), ("warmup", 2), ("profile", 1)];
const QUALIFICATION_ERRORS: [&str; 7] = [
    "hyperion",
    "foundation",
    "target",
    "context",
    "policy_logits",
    "reward_and_runtime",
    "cache_full_prefix",
];
const PREDICTION_BACKENDS: [&str; 3] = ["python", "rust", "cpp17"];
const BATCH_ROWS: [i128; 3] = [1, 32, 1024];
const REQUEST_COLUMNS: [&str; 6] = [
    "dataset",
    "mode",
    "seed",
    "elapsed_seconds",
    "accepted",
    "failure",
];
const BOUNDARY: &str = "Fitting data to first prediction, including preprocessing, encoders, policy, fitting, emission and compilation. Backends emitted sequentially; later milestones include earlier backend work. No LLM speed comparison.";

#[path = "inference_report/format.rs"]
mod format;
#[path = "inference_report/json.rs"]
mod json;
#[cfg(test)]
#[path = "inference_report/tests.rs"]
mod tests;

pub(super) use format::*;
pub(super) use json::*;

// ---------------------------------------------------------------------------------------------
// Journal inspection.

/// `Path.exists()`: absence (including a non-directory parent or a symlink loop) is false; any
/// other failure fails the report.
pub(super) fn exists(path: &Path) -> Result<bool> {
    match fs::metadata(path) {
        Ok(_) => Ok(true),
        // ENOENT, EBADF, ENOTDIR and ELOOP, and paths the OS cannot encode.
        Err(error)
            if matches!(error.raw_os_error(), Some(2 | 9 | 20 | 40))
                || error.kind() == std::io::ErrorKind::InvalidInput =>
        {
            Ok(false)
        }
        Err(error) => Err(error).with_context(|| format!("stat {}", path.display())),
    }
}

/// Directory entries whose names satisfy `matches`, or none when `dir` cannot be listed.
fn entries(dir: &Path, matches: impl Fn(&str) -> bool) -> Vec<(String, PathBuf)> {
    if !dir.is_dir() {
        return Vec::new();
    }
    let Ok(listing) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let Ok(listing) = listing.collect::<std::io::Result<Vec<_>>>() else {
        return Vec::new();
    };
    listing
        .into_iter()
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            matches(&name).then(|| (name, entry.path()))
        })
        .collect()
}

/// `journal.glob(prefix + '*/receipt.json')`, as (directory name, receipt path).
fn receipts(journal: &Path, prefix: &str) -> Vec<(String, PathBuf)> {
    entries(journal, |name| name.starts_with(prefix))
        .into_iter()
        .filter(|(_, dir)| dir.is_dir())
        .filter(|(_, dir)| !entries(dir, |name| name == "receipt.json").is_empty())
        .map(|(name, dir)| (name, dir.join("receipt.json")))
        .collect()
}

fn journal_path(journal: &Path, name: String) -> PathBuf {
    journal.join(name)
}

// ---------------------------------------------------------------------------------------------
// The report.

/// The CUDA qualification receipt is bound to this plan and every case passed within tolerance.
fn qualified(plan: &Json, q: &Json) -> Result<bool> {
    let cases = get_or(q, "cases", &EMPTY_ARRAY)?;
    let experiment = || index(plan, "experiment");
    let accepted = py_eq(at(q, "schema")?, index(plan, "schema")?)
        && is_true(get(q, "passed")?)
        && py_eq(
            at(q, "plan_identity")?,
            index(experiment()?, "methods_sha256")?,
        )
        && py_eq(
            at(q, "binary_sha256")?,
            index(index(plan, "binary")?, "sha256")?,
        )
        && py_eq(at(q, "source_sha")?, index(experiment()?, "source_sha")?)
        && py_len(cases)? == 8
        && distinct(
            py_iter(cases)?
                .iter()
                .map(|c| at(c, "case").cloned())
                .collect::<Result<_>>()?,
        )?
        .len()
            == 8
        && py_len(get_or(q, "devices", &EMPTY_ARRAY)?)? == 3
        && all_contain(index(q, "devices")?, "Cuda")?
        && py_contains(get_or(q, "policy_device", &EMPTY_STRING)?, "Cuda")?;
    if !accepted {
        return Ok(false);
    }
    for case in py_iter(cases)? {
        if !is_true(get(&case, "pass")?) {
            return Ok(false);
        }
        let names = distinct(py_iter(get_or(&case, "errors", &EMPTY_OBJECT)?)?)?;
        if !(names.len() == QUALIFICATION_ERRORS.len()
            && QUALIFICATION_ERRORS
                .iter()
                .all(|key| names.iter().any(|name| is(name, key))))
        {
            return Ok(false);
        }
        let Json::Object(errors) = index(&case, "errors")? else {
            bail!("AttributeError: qualification errors have no values()");
        };
        let within = errors.0.iter().all(|(_, error)| {
            number(error).is_some_and(|n| {
                n.is_finite()
                    && at_least_zero(n)
                    && compare(n, JsonNumber::Float(3e-4)).is_some_and(Ordering::is_le)
            })
        });
        if !within {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Planned direct-request slots per mode, frozen before execution; every slot needs one
/// terminal outcome. A plan without an explicit census uses the v2 study's.
fn census(plan: &Json) -> Result<(Json, Vec<(String, u64)>)> {
    let counts = match get(plan, "census")? {
        Some(counts) => counts.clone(),
        None => Json::Object(Object(
            DEFAULT_CENSUS
                .iter()
                .map(|(mode, count)| (mode.to_string(), Json::from(*count)))
                .collect(),
        )),
    };
    let modes = distinct(py_iter(&counts)?)?;
    let known = modes.len() == DEFAULT_CENSUS.len()
        && DEFAULT_CENSUS
            .iter()
            .all(|(mode, _)| modes.iter().any(|m| is(m, mode)));
    ensure!(known, "census modes or counts invalid");
    let Json::Object(members) = &counts else {
        bail!("AttributeError: census has no values()");
    };
    let mut planned = Vec::new();
    for (mode, count) in &members.0 {
        let count = match count {
            Json::Int(i) if *i >= 0 => u64::try_from(*i).context("census count overflow")?,
            Json::Bool(b) => u64::from(*b),
            _ => bail!("census modes or counts invalid"),
        };
        planned.push((mode.clone(), count));
    }
    Ok((counts, planned))
}

/// A whole-request receipt is bound to the plan, the slot, the CUDA devices and the timer.
fn request_valid(plan: &Json, d: &Json, r: &Json, mode: &str, seed: u64) -> Result<bool> {
    let experiment = || index(plan, "experiment");
    Ok(py_eq(at(r, "schema")?, index(plan, "schema")?)
        && py_eq(
            at(r, "plan_identity")?,
            index(experiment()?, "methods_sha256")?,
        )
        && py_eq(
            at(r, "binary_sha256")?,
            index(index(plan, "binary")?, "sha256")?,
        )
        && py_eq(at(r, "source_sha")?, index(experiment()?, "source_sha")?)
        && py_eq(at(r, "dataset")?, index(d, "id")?)
        && py_eq(at(r, "seed")?, &Json::from(seed))
        && is(at(r, "mode")?, mode)
        && py_eq(at(r, "input_sha256")?, index(index(d, "raw")?, "sha256")?)
        && is(at(r, "selected_origin")?, "neural_policy")
        && is(at(r, "ranking_backend")?, "candle_cuda_float32")
        && py_len(get_or(r, "devices", &EMPTY_ARRAY)?)? == 3
        && all_contain(index(r, "devices")?, "Cuda")?
        && py_contains(get_or(r, "policy_device", &EMPTY_STRING)?, "Cuda")?
        && is_true(get(
            get_or(r, "export", &EMPTY_OBJECT)?,
            "validation_in_timer",
        )?))
}

/// A completed backend milestone with verified parity, artifacts, bundle and commands.
fn backend_good(b: &Json, t: &Json) -> Result<bool> {
    let qualification = || get_or(b, "qualification", &EMPTY_OBJECT);
    Ok(is(at(b, "status")?, "completed")
        && number(t).is_some_and(|n| n.is_finite() && above_zero(n))
        && parity_verified(b)?
        && {
            let values = get_or(qualification()?, "values", &ZERO)?;
            number(values)
                .map(above_zero)
                .with_context(|| format!("TypeError: '>' on a JSON {}", kind(values)))?
        }
        && py_eq(
            at(qualification()?, "maximum_absolute_error")?,
            &Json::Int(0),
        )
        && all_files_verified(get_or(b, "files", &EMPTY_OBJECT)?)?
        && truthy(at(b, "files")?)
        && verified(get_or(b, "bundle", &EMPTY_OBJECT)?)?
        && all_commands_succeeded(get_or(b, "commands", &EMPTY_ARRAY)?)?)
}

fn all_files_verified(files: &Json) -> Result<bool> {
    let Json::Object(files) = files else {
        bail!("AttributeError: a JSON {} has no values()", kind(files));
    };
    for (_, pin) in &files.0 {
        if !verified(pin)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn all_commands_succeeded(commands: &Json) -> Result<bool> {
    for command in py_iter(commands)? {
        if !py_eq(at(&command, "exit_code")?, &Json::Int(0)) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn all_verified(pins: &Json) -> Result<bool> {
    for pin in py_iter(pins)? {
        if !verified(&pin)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn any_parity(backends: &Json) -> Result<bool> {
    for backend in py_iter(backends)? {
        if parity_verified(&backend)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Every returned prediction equals the qualification plan's first expected value bit for bit.
/// KeyError and TypeError mean false; other failures fail the report, as before.
fn batch_matches(r: &Json, a: &Json, prediction_plan: &Json) -> Result<bool> {
    let Some(data) = lookup(a, "data") else {
        return Ok(false);
    };
    let Ok(rows) = py_len(data) else {
        return Ok(false);
    };
    let Some(batch_rows) = lookup(r, "batch_rows") else {
        return Ok(false);
    };
    if !py_eq(&Json::from(rows), batch_rows) {
        return Ok(false);
    }
    for x in py_iter(data)? {
        let bits = pack(&x)?;
        let Some(expected) = lookup(prediction_plan, "expected").and_then(|e| lookup(e, "data"))
        else {
            return Ok(false);
        };
        let first = match expected {
            Json::Array(items) => items.first().cloned().context("IndexError: empty data")?,
            Json::Str(text) => Json::Str(
                text.chars()
                    .next()
                    .context("IndexError: empty data")?
                    .to_string(),
            ),
            _ => return Ok(false),
        };
        if bits != pack(&first)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// A retained prediction-API receipt: bound to the plan, the warm export's verified bundle and
/// qualification, exact bits, a planned backend and batch size and ten finite samples.
fn prediction_valid(
    plan: &Json,
    inputs: &[Json],
    journal: &Path,
    qualification: bool,
    r: &Json,
) -> Result<bool> {
    let dataset = at(r, "dataset")?.clone();
    let mut d = None;
    for candidate in inputs {
        if py_eq(index(candidate, "id")?, &dataset) {
            d = Some(candidate);
            break;
        }
    }
    let experiment = || index(plan, "experiment");
    let mut valid = qualification
        && match d {
            None => false,
            Some(d) => {
                py_eq(at(r, "source_sha")?, index(experiment()?, "source_sha")?)
                    && py_eq(
                        at(r, "plan_identity")?,
                        index(experiment()?, "methods_sha256")?,
                    )
                    && py_eq(
                        at(r, "binary_sha256")?,
                        index(index(plan, "binary")?, "sha256")?,
                    )
                    && py_eq(at(r, "input_sha256")?, index(index(d, "raw")?, "sha256")?)
                    && verified(get_or(r, "source_qualification", &EMPTY_OBJECT)?)?
                    && verified(get_or(r, "prediction", &EMPTY_OBJECT)?)?
                    && is(at(r, "parity")?, "exact_bits")
            }
        };
    let source =
        read(&journal_path(journal, format!("warm-{}-0", py_str(&dataset))).join("receipt.json"));
    let mut entry = EMPTY_OBJECT.clone();
    for b in py_iter(get_or(
        get_or(&source, "export", &EMPTY_OBJECT)?,
        "backends",
        &EMPTY_ARRAY,
    )?)? {
        if py_eq(at(&b, "backend")?, at(r, "backend")?) {
            entry = b;
            break;
        }
    }
    valid = valid
        && parity_verified(&entry)?
        && py_eq(
            at(r, "bundle_sha256")?,
            at(get_or(&entry, "bundle", &EMPTY_OBJECT)?, "sha256")?,
        )
        && py_eq(
            at(r, "source_qualification")?,
            at(&entry, "qualification_plan")?,
        );
    if valid {
        let prediction_plan = pinned(r, "source_qualification")?;
        let a = pinned(r, "prediction")?;
        valid = batch_matches(r, &a, &prediction_plan)?;
    }
    // Timing and backend membership are part of receipt validity.
    let backend = at(r, "backend")?;
    let batch_rows = at(r, "batch_rows")?;
    let samples = get_or(r, "samples_seconds", &EMPTY_ARRAY)?;
    Ok(valid
        && PREDICTION_BACKENDS.iter().any(|name| is(backend, name))
        && BATCH_ROWS.iter().any(|n| py_eq(batch_rows, &Json::Int(*n)))
        && py_len(samples)? == 10
        && py_iter(samples)?
            .iter()
            .all(|x| number(x).is_some_and(|n| n.is_finite() && at_least_zero(n))))
}

/// Summarize a timing journal against its plan: the report and each direct-request record.
fn summarize(plan: &Json, journal: &Path) -> Result<(Json, Vec<Json>)> {
    let q = read(&journal.join("qualification.json"));
    let qualification = qualified(plan, &q)?;
    let launch = read(&journal.join("launch.json"));
    let exit_code = get(&launch, "exit_code")?;
    let controller = object([
        (
            "status",
            if exit_code.is_some_and(|code| py_eq(code, &ZERO)) {
                "completed"
            } else if exit_code.is_some_and(|code| *code != Json::Null) {
                "failed"
            } else {
                "unknown"
            }
            .into(),
        ),
        ("launch", launch.clone()),
        (
            "qualification_status",
            if qualification {
                "passed"
            } else if truthy(&q) {
                "failed"
            } else {
                "not_executed_or_interrupted"
            }
            .into(),
        ),
    ]);
    let (mut records, mut missing, mut failures) = (Vec::new(), Vec::new(), Vec::new());
    let mut backend_times: Vec<(Json, Vec<Json>)> = Vec::new();
    let (counts, modes) = census(plan)?;
    let inputs = py_iter(index(plan, "inputs")?)?;
    let mut slots = Vec::new();
    for d in &inputs {
        for (mode, count) in &modes {
            for seed in 0..*count {
                slots.push(format!("{mode}-{}-{seed}", py_str(index(d, "id")?)));
            }
        }
    }
    for d in &inputs {
        for (mode, count) in &modes {
            for seed in 0..*count {
                let id = index(d, "id")?;
                let slot = journal_path(journal, format!("{mode}-{}-{seed}", py_str(id)));
                let p = slot.join("receipt.json");
                if !exists(&p)? {
                    let status = if exists(&slot)? {
                        "unexecuted_or_interrupted"
                    } else {
                        "unexecuted"
                    };
                    missing.push(object([
                        ("dataset", id.clone()),
                        ("mode", mode.as_str().into()),
                        ("seed", seed.into()),
                        ("status", status.into()),
                    ]));
                    continue;
                }
                let mut r = read(&p);
                let mut valid = request_valid(plan, d, &r, mode, seed)?;
                let elapsed = at(&r, "elapsed_seconds")?;
                valid = valid && number(elapsed).is_some_and(|n| n.is_finite() && at_least_zero(n));
                let export = get_or(&r, "export", &EMPTY_OBJECT)?;
                let recipe = get_or(export, "recipe", &EMPTY_OBJECT)?;
                valid = valid
                    && py_eq(
                        at(recipe, "original_proposal")?,
                        at(&r, "selected_program")?,
                    )
                    && verified(get_or(export, "fitted_output", &EMPTY_OBJECT)?)?;
                let mut completions = Vec::new();
                for b in py_iter(get_or(export, "backends", &EMPTY_ARRAY)?)? {
                    let t = at(&b, "first_prediction_seconds")?;
                    if qualification && valid && backend_good(&b, t)? {
                        let backend = index(&b, "backend")?;
                        completions.push(backend.clone());
                        if mode == "warm" {
                            hashable(id)?;
                            hashable(backend)?;
                            let key = Json::Array(vec![id.clone(), backend.clone()]);
                            match backend_times.iter_mut().find(|(k, _)| py_eq(k, &key)) {
                                Some((_, times)) => times.push(t.clone()),
                                None => backend_times.push((key, vec![t.clone()])),
                            }
                        }
                    } else if !is(at(&b, "status")?, "unsupported") {
                        failures.push(object([
                            ("dataset", id.clone()),
                            ("mode", mode.as_str().into()),
                            ("seed", seed.into()),
                            ("backend", at(&b, "backend")?.clone()),
                            ("status", at(&b, "status")?.clone()),
                            (
                                "reason",
                                get_or(
                                    &b,
                                    "reason",
                                    &Json::from("receipt or artifact validation failed"),
                                )?
                                .clone(),
                            ),
                        ]));
                    }
                }
                let accepted = qualification
                    && valid
                    && is_true(get(&r, "success")?)
                    && ["python", "rust"]
                        .iter()
                        .all(|x| completions.iter().any(|c| is(c, x)));
                assign(&mut r, "accepted", accepted.into())?;
                if !accepted {
                    failures.push(object([
                        ("dataset", id.clone()),
                        ("mode", mode.as_str().into()),
                        ("seed", seed.into()),
                        (
                            "failure",
                            get_or(&r, "failure", &Json::from("required path incomplete"))?.clone(),
                        ),
                    ]));
                }
                records.push(r);
            }
        }
    }
    let mut warm = Vec::new();
    for r in &records {
        if is(index(r, "mode")?, "warm") && truthy(index(r, "accepted")?) {
            warm.push(r);
        }
    }
    let mut gp = Vec::new();
    for d in &inputs {
        if matches!(get(d, "gp")?, None | Some(Json::Null)) {
            continue;
        }
        for arm in ["neural", "generic"] {
            let id = index(d, "id")?;
            let p = journal_path(journal, format!("gp-{}", py_str(id)))
                .join(arm)
                .join("receipt.json");
            let mut r = read(&p);
            if !truthy(&r) {
                r = object([
                    ("dataset", id.clone()),
                    ("arm", arm.into()),
                    ("status", "unexecuted".into()),
                ]);
            }
            ensure!(
                matches!(r, Json::Object(_)),
                "TypeError: a JSON {} receipt does not support item assignment",
                kind(&r)
            );
            let experiment = || index(plan, "experiment");
            let outcome = || get_or(&r, "outcome", &EMPTY_OBJECT);
            let accepted = qualification
                && py_eq(at(&r, "source_sha")?, index(experiment()?, "source_sha")?)
                && py_eq(
                    at(&r, "plan_identity")?,
                    index(experiment()?, "methods_sha256")?,
                )
                && py_eq(
                    at(&r, "binary_sha256")?,
                    index(index(plan, "binary")?, "sha256")?,
                )
                && py_eq(at(&r, "input_sha256")?, index(index(d, "raw")?, "sha256")?)
                && verified(get_or(outcome()?, "search", &EMPTY_OBJECT)?)?
                && truthy(at(outcome()?, "execution_inventory")?)
                && all_verified(index(index(&r, "outcome")?, "execution_inventory")?)?;
            let export_qualified = accepted
                && any_parity(get_or(
                    get_or(outcome()?, "export", &EMPTY_OBJECT)?,
                    "backends",
                    &EMPTY_ARRAY,
                )?)?;
            assign(&mut r, "accepted", accepted.into())?;
            assign(&mut r, "export_qualified", export_qualified.into())?;
            gp.push(r);
        }
    }
    let mut prediction = Vec::new();
    let mut paths = receipts(journal, "prediction-");
    paths.sort();
    for (_, path) in paths {
        let mut r = read(&path);
        let valid = prediction_valid(plan, &inputs, journal, qualification, &r)?;
        let accepted = valid && is(at(&r, "status")?, "completed");
        assign(&mut r, "accepted", accepted.into())?;
        prediction.push(r);
    }
    let mut seen = Vec::new();
    for r in &prediction {
        let key = [
            at(r, "dataset")?.clone(),
            at(r, "backend")?.clone(),
            at(r, "batch_rows")?.clone(),
        ];
        for member in &key {
            hashable(member)?;
        }
        seen.push(key);
    }
    for d in &inputs {
        let id = index(d, "id")?;
        let source =
            read(&journal_path(journal, format!("warm-{}-0", py_str(id))).join("receipt.json"));
        let mut entries: Vec<(Json, Json)> = Vec::new();
        for b in py_iter(get_or(
            get_or(&source, "export", &EMPTY_OBJECT)?,
            "backends",
            &EMPTY_ARRAY,
        )?)? {
            let key = at(&b, "backend")?.clone();
            hashable(&key)?;
            match entries.iter_mut().find(|(k, _)| py_eq(k, &key)) {
                Some(slot) => slot.1 = b,
                None => entries.push((key, b)),
            }
        }
        for backend in PREDICTION_BACKENDS {
            for size in BATCH_ROWS {
                hashable(id)?;
                if seen.iter().any(|[dataset, name, rows]| {
                    py_eq(dataset, id) && is(name, backend) && py_eq(rows, &Json::Int(size))
                }) {
                    continue;
                }
                let entry = entries
                    .iter()
                    .find(|(k, _)| is(k, backend))
                    .map_or(&EMPTY_OBJECT, |(_, b)| b);
                let status = if is(at(entry, "status")?, "unsupported") {
                    "unsupported"
                } else if exists(&journal_path(
                    journal,
                    format!("prediction-{}-{backend}-{size}", py_str(id)),
                ))? {
                    "interrupted"
                } else {
                    "unexecuted"
                };
                prediction.push(object([
                    ("dataset", id.clone()),
                    ("backend", backend.into()),
                    ("batch_rows", Json::Int(size)),
                    ("status", status.into()),
                    ("accepted", false.into()),
                ]));
            }
        }
    }
    // Direct receipts outside the frozen census are never silently ignored.
    let mut extra: Vec<String> = DEFAULT_CENSUS
        .iter()
        .flat_map(|(mode, _)| receipts(journal, &format!("{mode}-")))
        .map(|(name, _)| name)
        .filter(|name| !slots.contains(name))
        .collect();
    extra.sort();
    let warm_count = match index(&counts, "warm")? {
        Json::Int(i) => *i,
        Json::Bool(b) => i128::from(*b),
        other => bail!("census warm count is a JSON {}", kind(other)),
    };
    let planned_warm = warm_count * inputs.len() as i128;
    let accepted = records
        .iter()
        .map(|r| index(r, "accepted").map(truthy))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .filter(|a| *a)
        .count();
    let reconciled = records.len() + missing.len() == slots.len();
    let reconciliation = object([
        ("planned", slots.len().into()),
        ("receipts", records.len().into()),
        ("accepted", accepted.into()),
        ("rejected", (records.len() - accepted).into()),
        ("missing", missing.len().into()),
        (
            "unplanned_receipts",
            Json::Array(extra.iter().map(|name| name.as_str().into()).collect()),
        ),
        ("reconciled", reconciled.into()),
        (
            "failure_or_missing_rate",
            if slots.is_empty() {
                Json::Null
            } else {
                Json::Float((slots.len() - accepted) as f64 / slots.len() as f64)
            },
        ),
    ]);
    let mut failure = None;
    backend_times.sort_by(|a, b| {
        py_order(&a.0, &b.0).unwrap_or_else(|error| {
            failure.get_or_insert(error);
            Ordering::Equal
        })
    });
    if let Some(error) = failure {
        return Err(error);
    }
    let mut milestones = Vec::new();
    for (key, times) in &backend_times {
        let Json::Array(key) = key else {
            unreachable!("milestone keys are pairs")
        };
        let Json::Object(statistics) = distribution(times)? else {
            unreachable!("a distribution is an object")
        };
        let mut milestone = members([("dataset", key[0].clone()), ("backend", key[1].clone())]);
        milestone.0.extend(statistics.0);
        milestones.push(Json::Object(milestone));
    }
    let mut cold = entries(journal, |name| {
        name.len() >= "cold-process-.json".len()
            && name.starts_with("cold-process-")
            && name.ends_with(".json")
    });
    cold.sort();
    let elapsed = warm
        .iter()
        .map(|r| index(r, "elapsed_seconds").cloned())
        .collect::<Result<Vec<_>>>()?;
    let summary = object([
        ("schema", index(plan, "schema")?.clone()),
        ("controller", controller),
        ("qualified", qualification.into()),
        ("qualification", q.clone()),
        (
            "publication_headline",
            (qualification
                && planned_warm > 0
                && warm.len() as i128 == planned_warm
                && extra.is_empty()
                && reconciled)
                .into(),
        ),
        ("completed_successfully", warm.len().into()),
        ("expected", Json::Int(planned_warm)),
        (
            "census",
            object([
                ("modes", counts.clone()),
                (
                    "source",
                    if get(plan, "census")?.is_some() {
                        "plan"
                    } else {
                        "default-v2"
                    }
                    .into(),
                ),
            ]),
        ),
        ("reconciliation", reconciliation),
        ("missing", Json::Array(missing)),
        ("failures", Json::Array(failures)),
        (
            "warm_cumulative_backend_milestones",
            Json::Array(milestones),
        ),
        (
            "cold_observations",
            Json::Array(cold.iter().map(|(_, path)| read(path)).collect()),
        ),
        ("full_gp", Json::Array(gp)),
        ("prediction", Json::Array(prediction)),
        ("window", read(&journal.join("window.json"))),
        (
            "warm_complete_request_distribution",
            distribution(&elapsed)?,
        ),
        ("boundary", BOUNDARY.into()),
    ]);
    Ok((summary, records))
}

/// Write `summary.json`, `requests.csv` and `methodology.md` for a journal. `plan` holds the
/// plan bytes whose pin the caller has already verified.
pub(super) fn write(plan: &[u8], journal: &Path, output: &Path) -> Result<()> {
    let plan: Json = serde_json::from_slice(plan).context("plan JSON")?;
    let schema = index(&plan, "schema")?;
    hashable(schema)?;
    ensure!(
        SUPPORTED.iter().any(|supported| is(schema, supported)),
        "receipt version unsupported"
    );
    let (summary, records) = summarize(&plan, journal)?;
    fs::create_dir_all(output)?;
    fs::write(
        output.join("summary.json"),
        dumps(&summary, Style::REPORT)? + "\n",
    )?;
    let mut requests = String::new();
    let header: Vec<Json> = REQUEST_COLUMNS.iter().map(|c| Json::from(*c)).collect();
    csv_row(&header.iter().collect::<Vec<_>>(), &mut requests);
    for r in &records {
        let fields = REQUEST_COLUMNS
            .iter()
            .map(|column| get_or(r, column, &EMPTY_STRING))
            .collect::<Result<Vec<_>>>()?;
        csv_row(&fields, &mut requests);
    }
    fs::write(output.join("requests.csv"), requests)?;
    let reconciliation = index(&summary, "reconciliation")?;
    let methodology = format!(
        "# GPU timing study\n\n{BOUNDARY}\n\nQualified: {}. Completed warm Python and Rust paths: {}/{}. Planned direct slots: {}, accepted {}, missing {}, unplanned receipts {}. Missing attempts are unexecuted, never zero latency. Cold observations are individual process wall times. Full GP uses five fitting-side OOF folds,96 candidate slots and120seconds per arm; exports occur after the winner freezes. Prediction API samples exclude process initialization, which is reported as a separate residual containing I/O and teardown.\n",
        py_str(index(&summary, "qualified")?),
        py_str(index(&summary, "completed_successfully")?),
        py_str(index(&summary, "expected")?),
        py_str(index(reconciliation, "planned")?),
        py_str(index(reconciliation, "accepted")?),
        py_str(index(reconciliation, "missing")?),
        py_len(index(reconciliation, "unplanned_receipts")?)?,
    );
    fs::write(output.join("methodology.md"), methodology)?;
    Ok(())
}
