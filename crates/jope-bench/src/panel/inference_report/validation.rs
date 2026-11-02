use super::*;

pub(super) fn qualified(plan: &Json, q: &Json) -> Result<bool> {
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
pub(super) fn census(plan: &Json) -> Result<(Json, Vec<(String, u64)>)> {
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
pub(super) fn request_valid(
    plan: &Json,
    d: &Json,
    r: &Json,
    mode: &str,
    seed: u64,
) -> Result<bool> {
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
pub(super) fn backend_good(b: &Json, t: &Json) -> Result<bool> {
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

pub(super) fn all_files_verified(files: &Json) -> Result<bool> {
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

pub(super) fn all_commands_succeeded(commands: &Json) -> Result<bool> {
    for command in py_iter(commands)? {
        if !py_eq(at(&command, "exit_code")?, &Json::Int(0)) {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(super) fn all_verified(pins: &Json) -> Result<bool> {
    for pin in py_iter(pins)? {
        if !verified(&pin)? {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(super) fn any_parity(backends: &Json) -> Result<bool> {
    for backend in py_iter(backends)? {
        if parity_verified(&backend)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Every returned prediction equals the qualification plan's first expected value bit for bit.
/// KeyError and TypeError mean false; other failures fail the report, as before.
pub(super) fn batch_matches(r: &Json, a: &Json, prediction_plan: &Json) -> Result<bool> {
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
pub(super) fn prediction_valid(
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
