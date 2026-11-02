/// Summarize a timing journal against its plan: the report and each direct-request record.
use super::*;

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
